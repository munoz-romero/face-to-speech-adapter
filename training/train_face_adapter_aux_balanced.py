import os
import sys
import argparse
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from accelerate import Accelerator
from facenet_pytorch import InceptionResnetV1
import torchaudio
import soundfile as sf
import librosa

# Add StyleTTS2 to sys.path
sys.path.append('/home/voces/code/StyleTTS2')
try:
    from models import StyleEncoder
    from meldataset import preprocess
except ImportError:
    print("Error: Could not import StyleEncoder or preprocess from StyleTTS2. Make sure the path is correct.")
    sys.exit(1)

# Import Inference Utils
try:
    from inference_utils import StyleTTS2Inference
except ImportError:
    print("Warning: Could not import StyleTTS2Inference. Validation inference will be skipped.")
    StyleTTS2Inference = None

# -----------------------------------------------------------------------------
# 1. Dataset (LRS3)
# -----------------------------------------------------------------------------
class LRS3Dataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        """
        Custom Dataset for LRS3.
        Reads metadata from metadata_train_val.csv (or metadata_test.csv).

        Args:
            root_dir (str): Root directory of the dataset (containing metadata_*.csv).
            transform (callable, optional): Optional transform to be applied on the face image.
            split (str): 'train' or 'test'.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        if split == 'train':
            metadata_file = os.path.join(root_dir, 'metadata_train_validated.csv')
        elif split == 'val':
            metadata_file = os.path.join(root_dir, 'metadata_train_val_validated.csv')
        else: # test
            metadata_file = os.path.join(root_dir, 'metadata_test_validated.csv')

        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        # Load Face Attributes
        self.attributes = {}
        attr_file = os.path.join(root_dir, 'face_attributes.json')
        if os.path.exists(attr_file):
            with open(attr_file, 'r') as f:
                self.attributes = json.load(f)
        else:
            print(f"Warning: Attributes file not found at {attr_file}. Gender/Age supervision will be disabled (dummy values).")

        self._load_metadata(metadata_file)

    def _load_metadata(self, metadata_file):
        # Format: audio_path|text_phonemes|id_speaker
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    audio_path = parts[0]
                    # text = parts[1]
                    speaker_id = int(parts[2])

                    # Construct image path
                    # Replace /audio/ with /images/ and .wav with .jpg
                    img_path = audio_path.replace('/audio/', '/images/').replace('.wav', '.jpg')

                    self.samples.append({
                        'audio_path': audio_path,
                        'img_path': img_path,
                        'speaker_id': speaker_id
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Load Face Image
        img_path = sample['img_path']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image or handle error?
            # For training stability, maybe return a random noise or skip (but dataloader expects fixed size)
            # Let's generate noise (224x224 as per dataset spec)
            image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))

        if self.transform:
            image = self.transform(image)

        # 2. Load Audio/Mel Spectrogram
        # The file is a WAV. We need to compute Mel Spectrogram.
        # StyleTTS2 usually expects specific Mel parameters.
        # We'll use torchaudio to compute it on the fly.

        # StyleTTS2 params (from config.yml): sr=24000, n_fft=2048, win_length=1200, hop_length=300, n_mels=80

        audio_path = sample['audio_path']
        try:
            # Use soundfile instead of torchaudio.load to avoid backend issues
            wav_numpy, sr = sf.read(audio_path)

            # Handle channels (StyleTTS2 takes channel 0)
            if len(wav_numpy.shape) > 1:
                wav_numpy = wav_numpy[:, 0]

            # Resample if needed (Match StyleTTS2 logic using librosa)
            if sr != 24000:
                wav_numpy = librosa.resample(wav_numpy, orig_sr=sr, target_sr=24000)

            # Pad (Match StyleTTS2 _load_tensor)
            wav_numpy = np.concatenate([np.zeros([5000]), wav_numpy, np.zeros([5000])], axis=0)

            # Compute Mel using StyleTTS2's preprocess function
            # This ensures exact match of parameters (n_fft, hop, win, and implicit 16k sr bug if present)
            # preprocess returns [1, 80, T] and is already normalized ((log(mel)-mean)/std)
            mel = preprocess(wav_numpy)

            # mel shape: [1, 80, T]
            mel = mel.squeeze(0) # [80, T]

            # Random Crop
            target_len = 200
            if mel.size(1) > target_len:
                start = random.randint(0, mel.size(1) - target_len)
                mel = mel[:, start:start+target_len]
            else:
                # Pad
                pad_len = target_len - mel.size(1)
                mel = F.pad(mel, (0, pad_len))

        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            mel = torch.zeros(80, 200)

        # 3. Labels
        speaker_id = sample['speaker_id']

        # 4. Attributes (Gender/Age)
        # Default values if missing
        gender_label = 0.0 # Float for BCE
        age_label = 0.0 # Float for MSE

        if audio_path in self.attributes:
            attr = self.attributes[audio_path]
            gender_label = float(attr['gender']) # 1.0 or 0.0
            age_label = float(attr['age']) / 100.0 # Normalize 0-1 (approx)

        return image, mel, speaker_id, gender_label, age_label

# -----------------------------------------------------------------------------
# 2. Architecture: FaceToVoiceModel
# -----------------------------------------------------------------------------
class FaceToVoiceModel(nn.Module):
    def __init__(self, style_encoder_checkpoint, freeze_visual=True, soft_tuning=False):
        super().__init__()

        # A. Visual Encoder (FaceNet)
        # Pretrained on vggface2
        self.visual_encoder = InceptionResnetV1(pretrained='vggface2')

        # Freeze parameters
        if freeze_visual:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False

            if soft_tuning:
                # Unfreeze the last blocks (mixed_7a, repeat_3, block8, last_linear, logits)
                # InceptionResnetV1 structure: ... repeat_3, block8, avgpool, last_linear, last_bn
                for name, module in self.visual_encoder.named_children():
                    if any(x in name for x in ['mixed_7a', 'repeat_3', 'block8', 'last_linear', 'last_bn', 'logits']):
                         for param in module.parameters():
                            param.requires_grad = True

        # Output dimension of InceptionResnetV1 is 512 by default (if classify=False)
        self.d_vis = 512

        # B. Audio Encoder (Target - Frozen)
        # We need to instantiate StyleEncoder.
        # "timbre, 64-dim". So style_dim=64.
        # Standard StyleTTS2 config usually has dim_in=64 (channels), max_conv_dim=512.
        # We assume these defaults.
        self.audio_encoder = StyleEncoder(dim_in=64, style_dim=128, max_conv_dim=512)

        # Load Checkpoint
        if os.path.exists(style_encoder_checkpoint):
            print(f"Loading StyleEncoder weights from {style_encoder_checkpoint}")
            try:
                checkpoint = torch.load(style_encoder_checkpoint, map_location='cpu', weights_only=False)
            except TypeError:
                # Fallback for older PyTorch versions that don't support weights_only
                checkpoint = torch.load(style_encoder_checkpoint, map_location='cpu')

            # The checkpoint might be the full model or just the encoder.
            # If it's a full StyleTTS2 checkpoint, the keys will be like 'style_encoder.xxx'
            # We need to filter and load.
            if 'net' in checkpoint:
                state_dict = checkpoint['net']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Check if it's a nested dict (e.g. {'style_encoder': ...})
            if 'style_encoder' in state_dict and isinstance(state_dict['style_encoder'], dict):
                style_encoder_dict = state_dict['style_encoder']
                # Handle module. prefix if present (from DataParallel/DistributedDataParallel)
                if any(k.startswith('module.') for k in style_encoder_dict.keys()):
                    style_encoder_dict = {k.replace('module.', ''): v for k, v in style_encoder_dict.items()}
            else:
                # Try filtering keys (fallback for flat dicts)
                style_encoder_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('style_encoder.'):
                        style_encoder_dict[k.replace('style_encoder.', '')] = v
                    elif k.startswith('module.style_encoder.'):
                        style_encoder_dict[k.replace('module.style_encoder.', '')] = v

            # If still empty, maybe the checkpoint IS the style encoder (unlikely given the error)
            if not style_encoder_dict:
                 style_encoder_dict = state_dict

            # Now try to load
            try:
                self.audio_encoder.load_state_dict(style_encoder_dict, strict=True)
            except RuntimeError as e:
                # Check for spectral norm mismatch
                if 'weight_orig' in str(e):
                    print("Detected spectral norm mismatch. Removing spectral norm from StyleEncoder...")
                    from torch.nn.utils import remove_spectral_norm
                    for module in self.audio_encoder.modules():
                        try:
                            remove_spectral_norm(module)
                        except ValueError:
                            pass
                    # Try loading again
                    self.audio_encoder.load_state_dict(style_encoder_dict, strict=True)
                else:
                    print(f"Strict loading failed: {e}. Trying strict=False")
                    self.audio_encoder.load_state_dict(style_encoder_dict, strict=False)
            except Exception as e:
                print(f"Warning: Could not load StyleEncoder weights properly: {e}")
        else:
            print(f"Warning: Checkpoint {style_encoder_checkpoint} not found. Using random weights.")

        # Freeze Audio Encoder
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        self.audio_encoder.eval() # Set to eval mode

        self.d_aud = 128 # Target dimension (Timbre only)

        # C. Projector MLP (Trainable)
        # 512 -> 1024 -> 1024 -> 64
        self.projector_base = nn.Sequential(
            nn.Linear(self.d_vis, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.projector_head = nn.Sequential(
            nn.Linear(1024, self.d_aud)
            # Normalize will be applied in forward
        )

        # D. Auxiliary Heads
        self.gender_head = nn.Linear(self.d_aud, 1) # Logit output for BCE
        self.age_head = nn.Linear(self.d_aud, 1)    # Scalar output for MSE

    def forward_visual(self, images):
        # images: [B, 3, 160, 160]
        face_emb = self.visual_encoder(images) # [B, 512]

        # Projector
        hidden = self.projector_base(face_emb) # [B, 1024]

        # Main embedding
        proj_emb = self.projector_head(hidden) # [B, 128]

        # Aux outputs (using proj_emb to force latent space structure)
        pred_gender = self.gender_head(proj_emb)
        pred_age = self.age_head(proj_emb)

        proj_emb = F.normalize(proj_emb, p=2, dim=1)

        return proj_emb, pred_gender, pred_age

    def forward_audio(self, mels):
        # mels: [B, 80, T]

        with torch.no_grad():
            # StyleEncoder expects [B, 1, 80, T] because of Conv2d(1, ...)
            mels = mels.unsqueeze(1)

            # StyleEncoder forward returns style embedding
            # We assume it returns [B, 128]
            audio_emb = self.audio_encoder(mels)

            # Normalize
            audio_emb = F.normalize(audio_emb, p=2, dim=1)

        return audio_emb

# -----------------------------------------------------------------------------
# 3. Loss Functions: HybridContrastiveLoss
# -----------------------------------------------------------------------------
class HybridContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, lambda_nce=1.0, lambda_rkd=0.5, lambda_gender=0.2, lambda_age=0.1):
        super().__init__()
        self.temperature = temperature
        self.lambda_nce = lambda_nce
        self.lambda_rkd = lambda_rkd
        self.lambda_gender = lambda_gender
        self.lambda_age = lambda_age

        # Learnable temperature (optional, but good for CLIP-like training)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

        self.cross_entropy = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, proj_emb, audio_emb, labels, pred_gender=None, target_gender=None, pred_age=None, target_age=None):
        # proj_emb: [B, D] (Normalized)
        # audio_emb: [B, D] (Normalized)
        # labels: [B] (speaker_ids)

        # 1. InfoNCE (CLIP Loss) -> SupCon Loss
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits = logit_scale * torch.matmul(proj_emb, audio_emb.t()) # [B, B]

        # Máscara de Positivos (Basada en Speaker ID)
        # labels: [B] -> mask: [B, B] donde mask[i,j]=1 si son el mismo speaker
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(proj_emb.device)

        # Calcular Log-Softmax para estabilidad numérica
        # Expande la fórmula de SupCon: Suma sobre todos los positivos en el log
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach() # Estabilidad

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Calcular media de log-likelihood sobre los positivos
        # (mask * log_prob).sum(1) suma los log-probs de los positivos
        # mask.sum(1) es la cantidad de positivos por anchor (K)
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # Loss es el negativo de eso
        loss_nce = -mean_log_prob_pos.mean()

        # 2. RKD (Relational Knowledge Distillation) - Distance Wise
        # We want the distance structure in Visual space to match Audio space
        with torch.no_grad():
            d_audio = torch.cdist(audio_emb, audio_emb, p=2)
            d_audio = d_audio / (d_audio.mean() + 1e-8) # Normalize

        d_visual = torch.cdist(proj_emb, proj_emb, p=2)
        d_visual = d_visual / (d_visual.mean() + 1e-8) # Normalize

        loss_rkd = F.smooth_l1_loss(d_visual, d_audio)

        # 3. Auxiliary Losses
        loss_gender = torch.tensor(0.0, device=proj_emb.device)
        loss_age = torch.tensor(0.0, device=proj_emb.device)

        if pred_gender is not None and target_gender is not None:
            loss_gender = self.bce_loss(pred_gender.squeeze(), target_gender)

        if pred_age is not None and target_age is not None:
            loss_age = self.mse_loss(pred_age.squeeze(), target_age)

        # Total Loss
        total_loss = (self.lambda_nce * loss_nce) + \
                     (self.lambda_rkd * loss_rkd) + \
                     (self.lambda_gender * loss_gender) + \
                     (self.lambda_age * loss_age)

        return total_loss, {
            "loss_nce": loss_nce.item(),
            "loss_rkd": loss_rkd.item(),
            "loss_gender": loss_gender.item(),
            "loss_age": loss_age.item()
        }

def calculate_validation_metrics(proj_emb, audio_emb, temperature=0.07):
    """
    Calcula precisión de recuperación y similitud coseno promedio.
    proj_emb: [B, D] (Normalized)
    audio_emb: [B, D] (Normalized)
    """
    batch_size = proj_emb.size(0)

    # 1. Matriz de Similitud (Logits con temperatura para Accuracy)
    # [B, B]
    sim_matrix_logits = torch.matmul(proj_emb, audio_emb.t()) / temperature

    # 2. Etiquetas (La pareja correcta está en la diagonal)
    labels = torch.arange(batch_size).to(proj_emb.device)

    # 3. Top-1 Accuracy
    # Para cada cara (fila), ¿es la voz correcta (columna diagonal) la que tiene mayor valor?
    _, predicted_indices = torch.max(sim_matrix_logits, dim=1)
    correct_top1 = (predicted_indices == labels).sum().item()
    top1_acc = correct_top1 / batch_size

    # Top-5 Accuracy
    if batch_size >= 5:
        _, top5_indices = torch.topk(sim_matrix_logits, k=5, dim=1)
        correct_top5 = 0
        for i in range(batch_size):
            if labels[i] in top5_indices[i]:
                correct_top5 += 1
        top5_acc = correct_top5 / batch_size
    else:
        top5_acc = 1.0 if top1_acc == 1.0 else 0.0

    # 4. Average Latent Cosine Similarity (Diagonal mean, SIN temperatura)
    # Mide qué tan cerca están los vectores correctos en promedio
    raw_sim_matrix = torch.matmul(proj_emb, audio_emb.t())
    avg_cosine_sim = torch.diag(raw_sim_matrix).mean().item()

    return top1_acc, top5_acc, avg_cosine_sim

def calculate_sed(proj_emb):
    """
    Calcula la Speaker Embedding Diversity (SED).
    Mide la similitud media entre diferentes identidades en el lote.
    Un valor BAJO indica mayor diversidad (mejor).
    """
    # proj_emb ya debe estar normalizado (F.normalize)

    # 1. Matriz de Similitud [B, B]
    sim_matrix = torch.matmul(proj_emb, proj_emb.t())

    # 2. Máscara para ignorar la diagonal (auto-similitud) y duplicados (triángulo inferior)
    # triu(..., diagonal=1) selecciona solo lo que está POR ENCIMA de la diagonal principal
    batch_size = proj_emb.size(0)
    mask = torch.triu(torch.ones(batch_size, batch_size), diagonal=1).bool().to(proj_emb.device)

    # 3. Seleccionar valores y promediar
    # Si el batch es 1, no hay pares diversos, devolver 0 o nan
    if batch_size > 1:
        off_diagonal_sims = sim_matrix[mask]
        sed_score = off_diagonal_sims.mean().item()
    else:
        sed_score = 0.0

    return sed_score

# -----------------------------------------------------------------------------
# 4. Training Loop
# -----------------------------------------------------------------------------

class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, samples_per_class=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.num_classes = batch_size // samples_per_class

        # Crear índice: speaker_id -> [lista de índices en el dataset]
        self.labels = [s['speaker_id'] for s in dataset.samples]
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

        # Filtrar clases con menos muestras que las requeridas (opcional, o hacer sampling con reemplazo)
        self.classes = list(self.label_to_indices.keys())
        self.length = len(self.dataset) // self.batch_size

    def __iter__(self):
        for _ in range(self.length):
            # 1. Seleccionar P clases (speakers) aleatorios
            selected_classes = np.random.choice(self.classes, self.num_classes, replace=False)
            batch_indices = []

            for cls in selected_classes:
                # 2. Seleccionar K índices (muestras) de cada clase
                indices = self.label_to_indices[cls]
                # Si hay menos de K, permitir reemplazo
                replace = len(indices) < self.samples_per_class
                selected_indices = np.random.choice(indices, self.samples_per_class, replace=replace)
                batch_indices.extend(selected_indices)

            yield batch_indices

    def __len__(self):
        return self.length

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to LRS3 dataset')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to StyleTTS2 checkpoint (epoch_2nd_00020.pth)')
    parser.add_argument('--config_path', type=str, default='/home/voces/code/StyleTTS2/Models/LibriTTS/config.yml', help='Path to StyleTTS2 config')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='checkpoints_face_adapter')
    parser.add_argument('--test_faces_dir', type=str, default='test_faces', help='Directory with test face images for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to face adapter checkpoint to resume from')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start training from')
    parser.add_argument('--language', type=str, default='en-us', help='Language for phonemizer (e.g., en-us, es)')

    args = parser.parse_args()
    print(f"Arguments: {args}")

    accelerator = Accelerator(log_with="tensorboard", project_dir=args.output_dir)
    accelerator.init_trackers("face_adapter_logs", config=vars(args))

    os.makedirs(args.output_dir, exist_ok=True)

    # Data Augmentation
    train_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = LRS3Dataset(args.data_path, transform=train_transform, split='train')

    # Definir K (Samples per class)
    samples_per_class = 4 # K=4 es un buen estándar (4 vistas por persona)

    # Instanciar el Sampler
    train_sampler = BalancedBatchSampler(dataset, args.batch_size, samples_per_class=samples_per_class)

    # DataLoader con batch_sampler (OJO: shuffle debe ser False en el DataLoader porque el sampler ya mezcla)
    dataloader = DataLoader(dataset, batch_sampler=train_sampler, num_workers=args.num_workers)

    # Validation Dataset
    val_dataset = LRS3Dataset(args.data_path, transform=val_transform, split='val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # Model
    model = FaceToVoiceModel(args.checkpoint_path, freeze_visual=True, soft_tuning=True)

    # Resume from checkpoint if provided
    if args.resume_checkpoint is not None:
        print(f"Resuming training from checkpoint: {args.resume_checkpoint}")
        state_dict = torch.load(args.resume_checkpoint, map_location='cpu')
        # Use strict=False to allow loading weights from models without aux heads
        model.load_state_dict(state_dict, strict=False)

    # Loss
    criterion = HybridContrastiveLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(list(filter(lambda p: p.requires_grad, model.parameters())) + list(criterion.parameters()), lr=args.lr)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Prepare with Accelerator
    model, optimizer, dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, val_dataloader, scheduler
    )

    # Inference Engine (for validation)
    inference_engine = None
    if StyleTTS2Inference is not None and os.path.exists(args.test_faces_dir):
        try:
            inference_engine = StyleTTS2Inference(
                args.checkpoint_path,
                args.config_path,
                device=accelerator.device,
                language=args.language
            )
            accelerator.print(f"Inference engine initialized for validation (Language: {args.language}).")
        except Exception as e:
            import traceback
            traceback.print_exc()
            accelerator.print(f"Failed to initialize inference engine: {e}")

    best_val_top1 = 0.0

    # Training Loop
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        total_loss_epoch = 0

        for batch_idx, (images, mels, speaker_ids, gender, age) in enumerate(dataloader):
            # Move aux labels to device
            gender = gender.to(accelerator.device)
            age = age.to(accelerator.device)
            speaker_ids = speaker_ids.to(accelerator.device)

            # Fix Gender Label (Align with StyleTTS2 latent space if needed, or just standard BCE)
            # In previous script: target_gender_fixed = 1.0 - gender.float()
            target_gender_fixed = 1.0 - gender.float()

            # Forward Audio (Target)
            # Note: In distributed setting, we might need to gather audio_embs for contrastive loss across GPUs,
            # but for simplicity here we do it per-batch on each device.

            audio_emb = model.module.forward_audio(mels) if hasattr(model, 'module') else model.forward_audio(mels)

            # Forward Visual (Trainable)
            # Returns proj_emb, pred_gender, pred_age
            proj_emb, pred_gender, pred_age = model.module.forward_visual(images) if hasattr(model, 'module') else model.forward_visual(images)

            # Loss
            loss, loss_dict = criterion(proj_emb, audio_emb, speaker_ids, pred_gender, target_gender_fixed, pred_age, age.float())

            # Optimization
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            total_loss_epoch += loss.item()

            # Logging
            if batch_idx % 10 == 0:
                accelerator.print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f} (NCE: {loss_dict['loss_nce']:.4f}, RKD: {loss_dict['loss_rkd']:.4f}, Gender: {loss_dict['loss_gender']:.4f}, Age: {loss_dict['loss_age']:.4f})")
                accelerator.log({
                    "Train/loss": loss.item(),
                    "Train/nce": loss_dict['loss_nce'],
                    "Train/rkd": loss_dict['loss_rkd'],
                    "Train/loss_gender": loss_dict['loss_gender'],
                    "Train/loss_age": loss_dict['loss_age'],
                    "epoch": epoch + batch_idx / len(dataloader)
                }, step=epoch * len(dataloader) + batch_idx)

        scheduler.step()

        # --- Validation Loop ---
        model.eval()
        val_top1_acc = 0.0
        val_top5_acc = 0.0
        val_avg_cosine = 0.0
        val_sed = 0.0
        val_batches = 0

        accelerator.print("Running validation metrics...")
        with torch.no_grad():
            for val_batch in val_dataloader:
                # Unpack batch (handle potential variations in dataset return)
                if len(val_batch) == 5:
                    v_images, v_mels, _, _, _ = val_batch
                else:
                    # Fallback if dataset returns more items
                    v_images, v_mels = val_batch[0], val_batch[1]

                # Forward
                v_audio_emb = model.module.forward_audio(v_mels) if hasattr(model, 'module') else model.forward_audio(v_mels)
                v_proj_emb, _, _ = model.module.forward_visual(v_images) if hasattr(model, 'module') else model.forward_visual(v_images)

                # Metrics
                # Calculate effective temperature from logit_scale for metrics
                current_temp = 1.0 / criterion.logit_scale.exp().clamp(max=100).item()
                top1, top5, avg_cos = calculate_validation_metrics(v_proj_emb, v_audio_emb, current_temp)
                sed = calculate_sed(v_proj_emb)

                val_top1_acc += top1
                val_top5_acc += top5
                val_avg_cosine += avg_cos
                val_sed += sed
                val_batches += 1

        if val_batches > 0:
            val_top1_acc /= val_batches
            val_top5_acc /= val_batches
            val_avg_cosine /= val_batches
            val_sed /= val_batches

            accelerator.print(f"Validation Epoch {epoch+1}: Top-1 Acc: {val_top1_acc*100:.2f}%, Top-5 Acc: {val_top5_acc*100:.2f}%, Latent SECS: {val_avg_cosine:.4f}, SED: {val_sed:.4f}")
            accelerator.log({
                "Val/top1_acc": val_top1_acc,
                "Val/top5_acc": val_top5_acc,
                "Val/secs": val_avg_cosine,
                "Val/sed": val_sed
            }, step=epoch * len(dataloader))

            # Save Best Model
            if val_top1_acc > best_val_top1:
                best_val_top1 = val_top1_acc
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                save_path = os.path.join(args.output_dir, "best_model_top1.pth")
                accelerator.save(unwrapped_model.state_dict(), save_path)
                accelerator.print(f"New best model found (Top-1: {best_val_top1*100:.2f}%)! Saved to {save_path}")

        model.train()

        # Validation / Inference
        if inference_engine and (epoch + 1) % 5 == 0:
            accelerator.print("Running validation inference...")
            model.eval()
            with torch.no_grad():
                # Recursively find all images
                test_images = []
                for root, dirs, files in os.walk(args.test_faces_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                            test_images.append(os.path.join(root, file))

                for img_path in test_images:
                    # Create a relative name for logging/saving to avoid path issues but keep structure
                    rel_path = os.path.relpath(img_path, args.test_faces_dir)
                    safe_name = rel_path.replace(os.sep, '_')

                    try:
                        # Load and process image
                        img = Image.open(img_path).convert('RGB')
                        img_tensor = val_transform(img).unsqueeze(0).to(accelerator.device)

                        # Get embedding
                        proj_emb, _, _ = model.module.forward_visual(img_tensor) if hasattr(model, 'module') else model.forward_visual(img_tensor)

                        # Generate Audio
                        # text = "This is a generated audio from a face image."
                        text = ""
                        wav = inference_engine.inference(text, proj_emb)

                        # Save or Log
                        # Tensorboard audio logging
                        accelerator.trackers[0].writer.add_audio(f"val_audio/{safe_name}", wav, epoch, sample_rate=24000)

                        # Also save to disk
                        val_out_dir = os.path.join(args.output_dir, "val_samples", f"epoch_{epoch+1}")
                        os.makedirs(val_out_dir, exist_ok=True)
                        sf.write(os.path.join(val_out_dir, f"{safe_name}.wav"), wav, 24000)

                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        accelerator.print(f"Error inferring {safe_name}: {e}")
            model.train()

        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            save_path = os.path.join(args.output_dir, f"face_adapter_ep{epoch+1}.pth")
            accelerator.save(unwrapped_model.state_dict(), save_path)
            accelerator.print(f"Saved checkpoint to {save_path}")

    accelerator.end_training()

if __name__ == "__main__":
    main()

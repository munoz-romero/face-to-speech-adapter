"""
Script de inferencia con Gradio para Face Adapter (arquitectura Flexible).
Usa el modelo de train_face_adapter_flexible.py con el checkpoint de exp1_identity.
"""

import sys
import os
import torch
import gradio as gr
import numpy as np
import soundfile as sf
import yaml
from munch import Munch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1

# Add StyleTTS2 to path
sys.path.append('/home/voces/code/StyleTTS2')

from inference_utils import StyleTTS2Inference

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# lang = "en-us"  # English
lang = 'es'  # Spanish

# Paths
# English LibriTTS model
STYLETTS2_CONFIG = "/home/voces/code/StyleTTS2/Models/LibriTTS/config.yml"
STYLETTS2_CHECKPOINT = "/home/voces/code/StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth"
# Spanish Fonos model
STYLETTS2_CONFIG = "/home/voces/code/StyleTTS2/Models/Fonos/config_ft_es-ca_resume_v2.yml"
STYLETTS2_CHECKPOINT = "/home/voces/code/StyleTTS2/Models/Fonos/epoch_2nd_00026.pth"

# Face Adapter Checkpoint (exp1_identity - flexible architecture)
FACE_ADAPTER_CKPT = "/home/voces/datasets/face_project_models_exp/checkpoints_exp1_identity/best_model_top1.pth"
# FACE_ADAPTER_CKPT = "/home/voces/datasets/face_project_models_exp/checkpoints_exp1_identity/face_adapter_ep95.pth"
# FACE_ADAPTER_CKPT = "/home/voces/datasets/face_project_models_exp/checkpoints_exp1_identity/face_adapter_ep195.pth"

# Test faces directory
TEST_FACES_DIR = "/home/voces/code/face_project/test_faces"


# -----------------------------------------------------------------------------
# Model Architecture (from train_face_adapter_flexible.py)
# -----------------------------------------------------------------------------
class FaceToVoiceModel(nn.Module):
    def __init__(self, style_encoder_checkpoint, freeze_visual=True, soft_tuning=False):
        super().__init__()

        # A. Visual Encoder (FaceNet)
        self.visual_encoder = InceptionResnetV1(pretrained='vggface2')

        # Freeze parameters
        if freeze_visual:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False

            if soft_tuning:
                for name, module in self.visual_encoder.named_children():
                    if any(x in name for x in ['mixed_7a', 'repeat_3', 'block8', 'last_linear', 'last_bn', 'logits']):
                         for param in module.parameters():
                            param.requires_grad = True

        self.d_vis = 512

        # B. Audio Encoder (Target - Frozen) - Not needed for inference, but kept for compatibility
        from models import StyleEncoder
        self.audio_encoder = StyleEncoder(dim_in=64, style_dim=128, max_conv_dim=512)

        # Freeze Audio Encoder
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        self.audio_encoder.eval()

        # C. Projection Head (Adapter)
        self.projector_base = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.projector_head = nn.Linear(1024, 128)

        # D. Auxiliary Heads
        self.gender_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.age_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward_visual(self, images):
        # images: [B, 3, 160, 160]
        face_emb = self.visual_encoder(images)  # [B, 512]

        # Projector
        hidden = self.projector_base(face_emb)  # [B, 1024]

        # Main embedding
        proj_emb = self.projector_head(hidden)  # [B, 128]

        # Aux outputs
        pred_gender = self.gender_head(proj_emb)
        pred_age = self.age_head(proj_emb)

        proj_emb = F.normalize(proj_emb, p=2, dim=1)

        return proj_emb, pred_gender, pred_age


# -----------------------------------------------------------------------------
# Load Models
# -----------------------------------------------------------------------------
print(f"Loading StyleTTS2 from {STYLETTS2_CHECKPOINT}...")
inference_engine = StyleTTS2Inference(
    model_checkpoint=STYLETTS2_CHECKPOINT,
    config_path=STYLETTS2_CONFIG,
    device=device,
    language=lang
)

print(f"Loading Face Adapter from {FACE_ADAPTER_CKPT}...")

# Load adapter
try:
    ckpt = torch.load(FACE_ADAPTER_CKPT, map_location='cpu')
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    face_model = FaceToVoiceModel(
        style_encoder_checkpoint=STYLETTS2_CHECKPOINT,
        freeze_visual=True,
        soft_tuning=False
    )

    # Filter keys if necessary
    face_model.load_state_dict(state_dict, strict=False)
    print("Face Adapter loaded successfully.")
except Exception as e:
    print(f"Error loading Face Adapter: {e}")

face_model.to(device)
face_model.eval()


# -----------------------------------------------------------------------------
# Inference Function
# -----------------------------------------------------------------------------
def predict(image, text, ref_audio, alpha, beta, diffusion_steps, speed):
    if image is None:
        return None, "Por favor sube una imagen."

    # 1. Process Image
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    try:
        image_pil = Image.fromarray(image).convert('RGB')
        image_tensor = transform(image_pil).unsqueeze(0).to(device)
    except Exception as e:
        return None, f"Error procesando imagen: {e}"

    # 2. Get Face Embedding
    with torch.no_grad():
        out_visual = face_model.forward_visual(image_tensor)

        # Handle return type (Tuple: proj_emb, pred_gender, pred_age)
        if isinstance(out_visual, tuple):
            style_emb = out_visual[0]
        else:
            style_emb = out_visual
        # style_emb: [1, 128]

    # 3. Get Audio Reference (Prosody)
    ref_s = None
    if ref_audio is not None:
        try:
            sr, audio_data = ref_audio

            # Convert to tensor
            if audio_data.dtype == np.int16:
                audio_data = audio_data / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data / 2147483648.0
            elif audio_data.dtype == np.uint8:
                audio_data = (audio_data - 128) / 128.0

            audio_tensor = torch.from_numpy(audio_data).float()

            # Handle channels
            if audio_tensor.dim() > 1:
                audio_tensor = audio_tensor.mean(dim=1)

            audio_tensor = audio_tensor.unsqueeze(0)

            # Resample if needed
            if sr != 24000:
                resampler = torchaudio.transforms.Resample(sr, 24000)
                audio_tensor = resampler(audio_tensor)

            # Compute Mel
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=24000, n_fft=2048, win_length=1200, hop_length=300, n_mels=80,
                f_min=0, f_max=None, power=1.0, normalized=False
            ).to(device)

            audio_tensor = audio_tensor.to(device)
            mel = mel_transform(audio_tensor)
            mel = torch.log(torch.clamp(mel, min=1e-5))

            if mel.dim() == 2:
                mel = mel.unsqueeze(0)
            mel = mel.unsqueeze(1)

            # Encode Style
            with torch.no_grad():
                ref_s = inference_engine.model.style_encoder(mel)

        except Exception as e:
            print(f"Error processing reference audio: {e}")
            ref_s = None

    # 4. Run Inference
    try:
        wav = inference_engine.inference(
            text=text,
            style_emb=style_emb,
            ref_s=ref_s,
            alpha=alpha,
            beta=beta,
            diffusion_steps=diffusion_steps,
            embedding_scale=1.0,
            speed=speed
        )
    except Exception as e:
        return None, f"Error en inferencia: {e}"

    return (24000, wav), "Generado con éxito."


# -----------------------------------------------------------------------------
# Example Data
# -----------------------------------------------------------------------------
def get_example_faces():
    """Collect all test face images from test_faces directory."""
    faces = []
    for root, dirs, files in os.walk(TEST_FACES_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                faces.append(os.path.join(root, f))
    return sorted(faces)


EXAMPLE_TEXTS = [
    "Hola, soy una voz sintética generada a partir de una imagen de rostro.",
    "Buenos días, espero que estés teniendo un excelente día.",
    "La inteligencia artificial está revolucionando el mundo de la síntesis de voz.",
    "En un lugar de la Mancha, de cuyo nombre no quiero acordarme.",
    "El rápido zorro marrón salta sobre el perro perezoso.",
    "La tecnología nos permite crear voces únicas a partir de características faciales.",
    "Bienvenidos a esta demostración de mi trabajo fin de máster.",
    "Tu rostro contiene información que puede predecir cómo suena tu voz.",
]

example_faces = get_example_faces()

# Build example list: [image_path, text, ref_audio, alpha, beta, steps, speed]
EXAMPLES = []
for i, face_path in enumerate(example_faces[:8]):
    text = EXAMPLE_TEXTS[i % len(EXAMPLE_TEXTS)]
    EXAMPLES.append([
        face_path,
        text,
        None,
        0.5,
        1.0,
        25,
        1.0
    ])


# -----------------------------------------------------------------------------
# Gradio UI
# -----------------------------------------------------------------------------
theme = gr.themes.Base()

with gr.Blocks(theme=theme, title="Face to Voice (Flexible)") as demo:
    gr.Markdown("### Face to Voice - Flexible Model (Exp1 Identity)")

    with gr.Row():
        with gr.Column():
            inp_image = gr.Image(label="Rostro (Timbre)")
            inp_text = gr.Textbox(label="Texto", value="Hola, soy una voz sintética generada a partir de una imagen de rostro.", lines=3)
            inp_ref = gr.Audio(label="Referencia de Estilo (Opcional)", type="numpy")

            with gr.Accordion("Ajustes Avanzados", open=True):
                sld_alpha = gr.Slider(0, 1, value=0.5, label="Alpha (Identidad Rostro vs Ref)")
                sld_beta = gr.Slider(0, 1, value=1.0, label="Beta (Prosodia Difusión vs Ref)")
                sld_steps = gr.Slider(3, 50, value=25, step=1, label="Pasos de Difusión")
                sld_speed = gr.Slider(0.5, 2.0, value=1.0, label="Velocidad")

            btn_gen = gr.Button("Generar Voz", variant="primary")

        with gr.Column():
            out_audio = gr.Audio(label="Resultado")
            out_msg = gr.Textbox(label="Estado")

    gr.Examples(
        examples=EXAMPLES,
        inputs=[inp_image, inp_text, inp_ref, sld_alpha, sld_beta, sld_steps, sld_speed],
        outputs=[out_audio, out_msg],
        fn=predict,
        cache_examples=False
    )

    btn_gen.click(predict, [inp_image, inp_text, inp_ref, sld_alpha, sld_beta, sld_steps, sld_speed], [out_audio, out_msg])

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7862)

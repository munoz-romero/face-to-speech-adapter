import os
import sys
import torch
import torchaudio
import numpy as np
import yaml
from munch import Munch
from PIL import Image
from torchvision import transforms
import soundfile as sf
import phonemizer

# Add StyleTTS2 to sys.path
sys.path.append('/home/voces/code/StyleTTS2')

try:
    from models import *
    from utils import *
    from text_utils import TextCleaner
    from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
except ImportError:
    print("Error: Could not import StyleTTS2 modules. Make sure the path is correct.")

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

class StyleTTS2Inference:
    def __init__(self, model_checkpoint, config_path, device='cuda', language='en-us'):
        self.device = device
        self.config = yaml.safe_load(open(config_path))

        # Load Model
        self.model = self._load_model(model_checkpoint)
        # self.model is a Munch object (dict-like), so we cannot call .to() on it directly.

        # Load Sampler
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False
        )

        self.text_cleaner = TextCleaner()
        self.language = language

        # Initialize Phonemizer
        try:
            import phonemizer
            self.global_phonemizer = phonemizer.backend.EspeakBackend(
                language=language,
                preserve_punctuation=True,
                with_stress=True
            )
        except RuntimeError:
            print("Warning: espeak not found. Text-to-phoneme conversion will not work.")
            self.global_phonemizer = None
        except ImportError:
            print("Warning: phonemizer not found. Text-to-phoneme conversion will not work.")
            self.global_phonemizer = None

    def _load_model(self, checkpoint_path):

        from Utils.ASR.models import ASRCNN
        from Utils.JDC.model import JDCNet
        from Utils.PLBERT.util import load_plbert

        # Paths (Hardcoded based on typical StyleTTS2 setup or config)
        ASR_config = self.config.get('ASR_config', 'Utils/ASR/config.yml')
        ASR_path = self.config.get('ASR_path', 'Utils/ASR/epoch_00080.pth')
        F0_path = self.config.get('F0_path', 'Utils/JDC/bst.t7')
        BERT_path = self.config.get('PLBERT_dir', 'Utils/PLBERT/')

        # Fix paths to be absolute if needed
        base_path = '/home/voces/code/StyleTTS2'
        def fix_path(p):
            if not os.path.isabs(p):
                return os.path.join(base_path, p)
            return p

        ASR_config = fix_path(ASR_config)
        ASR_path = fix_path(ASR_path)
        F0_path = fix_path(F0_path)
        BERT_path = fix_path(BERT_path)

        # Load ASR
        # Config from Utils/ASR/config.yml says hidden_dim=256, token_embedding_dim=512
        text_aligner = ASRCNN(input_dim=80, hidden_dim=256, n_token=178, n_layers=6, token_embedding_dim=512)

        try:
            checkpoint = torch.load(ASR_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(ASR_path, map_location='cpu')

        text_aligner.load_state_dict(checkpoint['model'])
        text_aligner.to(self.device)
        text_aligner.eval()

        # Load F0
        pitch_extractor = JDCNet(num_class=1, seq_len=192)
        try:
            checkpoint = torch.load(F0_path, map_location='cpu', weights_only=False)['net']
        except TypeError:
            checkpoint = torch.load(F0_path, map_location='cpu')['net']

        pitch_extractor.load_state_dict(checkpoint)
        pitch_extractor.to(self.device)
        pitch_extractor.eval()

        # Load BERT
        plbert = load_plbert(BERT_path)
        plbert.to(self.device)
        plbert.eval()

        # Build Model
        model_params = recursive_munch(self.config['model_params'])
        model = build_model(model_params, text_aligner, pitch_extractor, plbert)

        # Load Checkpoint
        _ = [model[key].eval() for key in model]
        _ = [model[key].to(self.device) for key in model]

        try:
            params = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            params = torch.load(checkpoint_path, map_location='cpu')

        if 'net' in params:
            params = params['net']
        elif 'model' in params:
            params = params['model']

        for key in model:
            if key in params:
                state_dict = params[key]
                # Handle module. prefix if present
                if any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                try:
                    model[key].load_state_dict(state_dict)
                except RuntimeError as e:
                    if 'weight_orig' in str(e):
                        print(f"Detected spectral norm mismatch in {key}. Removing spectral norm...")
                        from torch.nn.utils import remove_spectral_norm
                        for module in model[key].modules():
                            try:
                                remove_spectral_norm(module)
                            except ValueError:
                                pass
                        try:
                            model[key].load_state_dict(state_dict)
                            print(f"Successfully loaded {key} after removing spectral norm.")
                        except Exception as e2:
                            print(f"Could not load {key} even after removing spectral norm: {e2}")
                    else:
                        print(f"Could not load {key}: {e}")
                except Exception as e:
                    print(f"Could not load {key}: {e}")

        return model

    def inference(self, text, style_emb, ref_s=None, output_path=None, alpha=0.1, beta=0.9, diffusion_steps=20, embedding_scale=1.0, speed=1.0):
        """
        text: str
        style_emb: torch.Tensor [1, 64] (Timbre only) or [1, 128]
        ref_s: torch.Tensor [1, 128] (Prosody Reference) - Optional
        alpha: float (0.0 - 1.0). Weight of diffusion for Timbre.
               Low value (e.g. 0.1) keeps input identity. High value (e.g. 0.9) samples new identity.
        beta: float (0.0 - 1.0). Weight of diffusion for Prosody.
               High value (e.g. 0.9) samples new prosody (good if input prosody is dummy).
        speed: float. Speed factor (default 1.0).
        """
        # Preprocess text
        if text is None or text.strip() == "":
             # Default text for validation
             text = "lˈɪɾəl ɡɹˈiːn tˈoʊd hˌuːz lˈɛɡ dʌθ twˈɪst, ɡˌoʊ tə ðə kˈɔːɹnɚ ʌvwˈɪtʃ juː wˈɪst, ænd bɹˈɪŋ tə mˌiː ðə lˈɑːɹdʒ ˈoʊld kˈɪst."
             if self.language == 'es':
                 text = "el βelˈoθ muɾθjˈelaɣo indˈu komˈia felˈiθ kaɾðˈiʎo i kˈiwi. la θˌiɣuˈeɲa tokˈaβa el sˌaksofˈon detɾˈas ðel palˈɛnke ðe pˈaxa."
        else:
             # Phonemize text
             if self.global_phonemizer is None:
                 # Check if text looks like phonemes (simple heuristic or just assume user knows what they are doing if espeak is missing)
                 # Or raise error
                 print("Warning: espeak not available, assuming input text is already phonemized.")
             else:
                 text = text.strip()
                 text = text.replace('"', '')
                 text = self.global_phonemizer.phonemize([text])[0]

        tokens = self.text_cleaner(text)
        tokens.insert(0, 0) # Add start token if needed, StyleTTS2 usually does this in inference
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)

            # Style Embedding Handling
            # The model expects style_emb of shape [B, 256] (Timbre 128 + Prosody 128)
            # Our face adapter returns [B, 128] (Timbre).

            if ref_s is not None:
                # If reference audio style is provided, use it for prosody
                if ref_s.dim() == 1: ref_s = ref_s.unsqueeze(0)
                style_emb = torch.cat([style_emb, ref_s], dim=-1) # [1, 256]
            elif style_emb.shape[-1] == 128:
                # Fallback: duplicate timbre for prosody
                style_emb = torch.cat([style_emb, style_emb], dim=-1) # [1, 256]

            if hasattr(self.model, 'inference'):
                wav = self.model.inference(tokens, style_emb, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale, speed=speed)
            else:
                wav = self._inference_routine(tokens, style_emb, diffusion_steps, embedding_scale, alpha, beta, speed=speed)

        if output_path:
            sf.write(output_path, wav, 24000)

        return wav

    def _inference_routine(self, x, style_emb, diffusion_steps=5, embedding_scale=1, alpha=0.3, beta=0.7, speed=1.0):
        # x: tokens [1, N]
        # style_emb: [1, 128]

        # This is a simplified version of the inference logic found in StyleTTS2 demos

        # 1. BERT
        input_lengths = torch.LongTensor([x.shape[-1]]).to(self.device)
        text_mask = length_to_mask(input_lengths).to(self.device)
        text_mask = text_mask.bool() # Force boolean

        t_en = self.model.text_encoder(x, input_lengths, text_mask)

        # Debug info
        # print(f"text_mask dtype: {text_mask.dtype}")

        bert_emb = self.model.bert(x, attention_mask=(~text_mask).int())
        d_en = self.model.bert_encoder(bert_emb).transpose(-1, -2)

        # 2. Duration
        s_pred = self.sampler(noise = torch.randn(1, 256).unsqueeze(1).to(self.device),
                              embedding=bert_emb,
                              embedding_scale=embedding_scale,
                              features=style_emb, # reference from the same speaker as the embedding
                              num_steps=diffusion_steps).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * style_emb[:, :128]
        s = beta * s + (1 - beta)  * style_emb[:, 128:]

        d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x_p, _ = self.model.predictor.lstm(d)
        duration = self.model.predictor.duration_proj(x_p)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        dur_data = pred_dur.sum().data

        pred_aln_trg = torch.zeros(input_lengths.item(), int(dur_data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))

        # Decoder type check (assuming hifigan or istftnet)
        # if model_params.decoder.type == "hifigan": ...
        # For simplicity, we assume standard behavior or check config
        if self.config['model_params']['decoder']['type'] == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
        if self.config['model_params']['decoder']['type'] == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()[..., :-50]

        # F0
        f0_pred = self.model.predictor.f0_decoder(d, asr_pred, s)

        # N_Mels
        m_pred = self.model.decoder(d, f0_pred, s, ref)

        return m_pred.squeeze().cpu().numpy()

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d

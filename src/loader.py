from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion

import model_converter
import vae


def preload_models_from_standard_weights(ckpt_path, vae_checkpoint_path=None, attention_type="xformers", device="cuda"):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    if vae_checkpoint_path:
        encoder, decoder = vae.load_vae_models(vae_checkpoint_path, device)

    else:
        encoder = VAE_Encoder().to(device)
        encoder.load_state_dict(state_dict["encoder"], strict=True)

        decoder = VAE_Decoder().to(device)
        decoder.load_state_dict(state_dict["decoder"], strict=True)

    diffusion = Diffusion(attention_type=attention_type).to(device)
    diffusion.load_state_dict(state_dict["diffusion"], strict=True)

    clip = CLIP(attention_type=attention_type).to(device)
    clip.load_state_dict(state_dict["clip"], strict=True)

    return {
        "clip": clip,
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": diffusion,
    }
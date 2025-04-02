from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion

import model_converter

def load_vae_models(ckpt_path, device="cuda"):
    checkpoint = torch.load(ckpt_path, map_location=device)

    encoder = VAE_Encoder().to(device)
    decoder = VAE_Decoder().to(device)

    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    decoder.load_state_dict(checkpoint["decoder"], strict=True)

    return encoder, decoder

def preload_models_from_standard_weights(ckpt_path, vae_checkpoint_path=None, device="cuda"):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    if vae_checkpoint_path:
        encoder, decoder = load_vae_models(vae_checkpoint_path, device)

    else:
        encoder = VAE_Encoder().to(device)
        encoder.load_state_dict(state_dict["encoder"], strict=True)

        decoder = VAE_Decoder().to(device)
        decoder.load_state_dict(state_dict["decoder"], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict["diffusion"], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict["clip"], strict=True)

    return {
        "clip": clip,
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": diffusion,
    }
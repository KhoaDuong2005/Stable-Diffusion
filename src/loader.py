from clip import CLIP
from diffusion import Diffusion

import model_converter

def convert_vae_state_dict(state_dict):
    """Convert old VAE state dict format to new format with encoder_layers/decoder_layers structure"""
    new_state_dict = {}
    
    # Process encoder state dict
    if "encoder" in state_dict:
        encoder_dict = {}
        for k, v in state_dict["encoder"].items():
            if k.startswith("0") or k.startswith("1") or k.startswith("2"):
                # Extract the numeric index and the rest of the key
                parts = k.split(".", 1)
                if len(parts) == 1:  # Handle keys like "0.weight"
                    index = int(parts[0])
                    rest = "weight" if "weight" in k else "bias"
                else:  # Handle keys like "1.groupnorm_1.weight"
                    index = int(parts[0])
                    rest = parts[1]
                
                new_key = f"encoder_layers.{index}.{rest}"
                encoder_dict[new_key] = v
            elif k.startswith("final_norm") or k.startswith("final_layers"):
                # These keys are already correct
                encoder_dict[k] = v
            else:
                # Pass through any other keys
                encoder_dict[k] = v
                
        state_dict["encoder"] = encoder_dict
    
    # Process decoder state dict similarly
    if "decoder" in state_dict:
        decoder_dict = {}
        for k, v in state_dict["decoder"].items():
            if k.isdigit() or (k[0].isdigit() and "." in k):
                parts = k.split(".", 1)
                if len(parts) == 1:
                    index = int(parts[0])
                    rest = "weight" if "weight" in k else "bias"
                else:
                    index = int(parts[0])
                    rest = parts[1]
                
                new_key = f"decoder_layers.{index}.{rest}"
                decoder_dict[new_key] = v
            elif k.startswith("final_norm") or k.startswith("final_layers"):
                decoder_dict[k] = v
            else:
                decoder_dict[k] = v
                
        state_dict["decoder"] = decoder_dict
    
    return state_dict

TESTING = False

def load_vae_models(vae_checkpoint_path, device="cuda"):
    try:
        checkpoint = torch.load(vae_checkpoint_path, map_location=device, weights_only=True)
    except Exception as e:
        try:
            import torch.serialization
            from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
            
            with torch.serialization.safe_globals([ModelCheckpoint]):
                checkpoint = torch.load(vae_checkpoint_path, map_location=device)
        except Exception:
            print("Warning: Loading checkpoint with weights_only=False. Only do this if you trust the source of this file.")
            checkpoint = torch.load(vae_checkpoint_path, map_location=device, weights_only=False)
    

    
    if TESTING:
        from encoder import VAE_Encoder_Optimized as VAE_Encoder
        from decoder import VAE_Decoder_Optimized as VAE_Decoder
    else:
        from encoder import VAE_Encoder
        from decoder import VAE_Decoder
    
    encoder = VAE_Encoder().to(device)
    decoder = VAE_Decoder().to(device)
    
    if "encoder" in checkpoint and "decoder" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder"], strict=False)  
        decoder.load_state_dict(checkpoint["decoder"], strict=False)
    elif "state_dict" in checkpoint:
        encoder_dict = {k.replace("encoder.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("encoder.")}
        decoder_dict = {k.replace("decoder.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("decoder.")}
        
        encoder.load_state_dict(encoder_dict, strict=False) 
        decoder.load_state_dict(decoder_dict, strict=False) 
    else:
        try:
            if "conv_in.weight" in checkpoint and "conv_out.weight" in checkpoint:
                print("Detected standard SD VAE format, attempting conversion...")
                
                # Use the convert function to map OrangeMix VAE weights
                from model_converter import convert_vae_state_dict
                
                encoder_state_dict, decoder_state_dict = convert_vae_state_dict(checkpoint)
                
                encoder.load_state_dict(encoder_state_dict, strict=False)
                decoder.load_state_dict(decoder_state_dict, strict=False)
                
                print("VAE conversion completed with non-strict loading")
            else:
                print("Warning: Using default weights since checkpoint format doesn't match model structure")
        except Exception as e:
            print(f"Error during VAE conversion: {e}")
    
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder


class VAE:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x, noise=None):
        return self.encoder(x, noise)

    def decode(self, z):
        return self.decoder(z)

def preload_models_from_standard_weights(ckpt_path, vae_checkpoint_path=None, attention_type="xformers", device="cuda"):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    if vae_checkpoint_path:
        encoder, decoder = load_vae_models(vae_checkpoint_path, device)

    else:
        if TESTING:
            from encoder import VAE_Encoder_Optimized as VAE_Encoder
            from decoder import VAE_Decoder_Optimized as VAE_Decoder
        else:
            from encoder import VAE_Encoder
            from decoder import VAE_Decoder
        encoder = VAE_Encoder().to(device)
        encoder.load_state_dict(state_dict["encoder"], strict=True)

        decoder = VAE_Decoder().to(device)
        decoder.load_state_dict(state_dict["decoder"], strict=True)
        
    vae_model = VAE(encoder, decoder)

    diffusion = Diffusion(attention_type=attention_type).to(device)
    diffusion.load_state_dict(state_dict["diffusion"], strict=True)

    clip = CLIP(attention_type=attention_type).to(device)
    clip.load_state_dict(state_dict["clip"], strict=True)

    return {
        "clip": clip,
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": diffusion,
        "vae": vae_model,
    }
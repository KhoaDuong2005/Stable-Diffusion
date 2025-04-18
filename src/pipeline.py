import torch
import numpy as np
from tqdm import tqdm
from sampler.ddpm import DDPMSampler
from sampler.euler import EulerSampler
from PIL import Image
from IPython.display import clear_output, display

WIDTH = 512
HEIGHT = 512

LATENTS_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8

def generate(
    prompt: str,
    negative_prompt: str,
    input_image=None,
    strength=0.7, 
    do_cfg=True, 
    cfg_scale=7, 
    sampler="ddpm", 
    steps=20, 
    models=(), 
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
    fp16_enabled=False,
):
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("Strength must be between 0 and 1")
        
        # Define to_idle: moves a module to the idle_device if provided, else returns it unchanged.
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        # If seed is None, generator will be random

        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # Convert the prompt into tokens.
            pos_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            pos_tokens = torch.tensor(pos_tokens, dtype=torch.long, device=device)
            pos_context = clip(pos_tokens)

            neg_tokens = tokenizer.batch_encode_plus(
                [negative_prompt], padding="max_length", max_length=77
            ).input_ids
            neg_tokens = torch.tensor(neg_tokens, dtype=torch.long, device=device)
            neg_context = clip(neg_tokens)

            # Concatenate positive and negative contexts along the batch dimension.
            context = torch.cat([pos_context, neg_context])
        else:
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)

        to_idle(clip)

        sampler_class_name_upper = f"{sampler.upper()}Sampler"
        sampler_class_name_capital = f"{sampler.capitalize()}Sampler"

        sampler_class = None

        if sampler_class_name_upper in globals():
            sampler_class = globals()[sampler_class_name_upper]
        elif sampler_class_name_capital in globals():
            sampler_class = globals()[sampler_class_name_capital]
        else:
            raise ValueError(f"Sampler '{sampler}' not found")

        print(f"Using {sampler_class.__name__} sampler")

        sampler = sampler_class(generator)
        sampler.set_steps(steps)
        print(f"Using {sampler.__class__.__name__} with {steps} steps")


        latents_shape = (1, 4, LATENT_HEIGHT, LATENTS_WIDTH)

        if input_image:  # img2img branch
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # Convert H, W, C to batch, C, H, W
            input_image_tensor = input_image_tensor.unsqueeze(0).permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            to_idle(encoder)
        else:  # txt2img branch
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        
        decoder = models["decoder"]
        decoder.to(device)

        # Iterate through timesteps to denoise the latents.
        for i, timestep in enumerate(tqdm(sampler.timesteps)):
            if fp16_enabled:
                print("Using FP16 for inference")
                scalar = torch.amp.GradScaler("cuda")
                with torch.amp.autocast(device_type=device, dtype=torch.float16):
                    time_embedding = get_time_embedding(timestep).to(device)
                    model_input = latents
                    if do_cfg:
                        # Duplicate batch dimension for CFG.
                        model_input = model_input.repeat(2, 1, 1, 1)
                        # Expect diffusion() to return a tuple (output_pos, output_neg)
                        output_pos, output_neg = diffusion(model_input, context, time_embedding)
                        model_output = cfg_scale * (output_pos - output_neg) + output_neg
                    else:
                        model_output = diffusion(model_input, context, time_embedding)
                    
                    latents = sampler.step(timestep, latents, model_output)
            

            else:
                
                time_embedding = get_time_embedding(timestep).to(device)
                model_input = latents
                if do_cfg:
                    model_input = model_input.repeat(2, 1, 1, 1)
                    output_pos, output_neg = diffusion(model_input, context, time_embedding)
                    model_output = cfg_scale * (output_pos - output_neg) + output_neg
                else:
                    model_output = diffusion(model_input, context, time_embedding)
                
                latents = sampler.step(timestep, latents, model_output)

            # Decode and display the intermediate image
            intermediate_latents = latents.clone()  # Clone latents to avoid in-place modification issues
            intermediate_image = decoder(intermediate_latents)
            intermediate_image = rescale(intermediate_image, (-1, 1), (0, 255), clamp=True)
            intermediate_image = intermediate_image.permute(0, 2, 3, 1)
            intermediate_image = intermediate_image.to("cpu", torch.uint8).numpy()[0]
            
            # Clear previous output and display the image in Jupyter Notebook
            clear_output(wait=True)
            display(Image.fromarray(intermediate_image))

        to_idle(diffusion)
        to_idle(decoder)
        
        torch.cuda.empty_cache()

        return intermediate_image

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range

    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    
    if clamp:
        x = torch.clamp(x, new_min, new_max)
    return x


def get_time_embedding(timestep):
    # Create time embedding vector of shape (1, 320)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


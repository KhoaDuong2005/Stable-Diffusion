import re
import os
import torch
from safetensors.torch import load_file

class LoRAHandler:
    def __init__(self, base_path="loras"):
        self.base_path = base_path
        self.lora_pattern = re.compile(r"<lora:(.*?):(.*?)>")
        self.loaded_loras = {}
        os.makedirs(self.base_path, exist_ok=True)
        print(f"LoRA directory: {os.path.abspath(self.base_path)}")

    def extract_loras(self, prompt):
        matches = self.lora_pattern.findall(prompt)
        loras = [(name, float(weight)) for name, weight in matches]
        clean_prompt = self.lora_pattern.sub("", prompt).strip()
        if loras:
            print(f"Found LoRAs in prompt: {loras}")
        return clean_prompt, loras

    def load_lora(self, name):
        if name in self.loaded_loras:
            return self.loaded_loras[name]
        for ext in [".safetensors", ".pt", ".bin"]:
            lora_path = os.path.join(self.base_path, f"{name}{ext}")
            if os.path.exists(lora_path):
                print(f"Loading LoRA: {lora_path}")
                try:
                    if ext == ".safetensors":
                        lora_weights = load_file(lora_path)
                    else:
                        lora_weights = torch.load(lora_path, map_location="cpu")
                    print(f"LoRA loaded with {len(lora_weights)} keys")
                    processed = {
                        'text': [],
                        'unet': [],
                        'all': []
                    }
                    for key in lora_weights.keys():
                        if ".lora_up.weight" in key:
                            base_key = key.replace(".lora_up.weight", "")
                            down_key = key.replace("lora_up", "lora_down")
                            if down_key in lora_weights:
                                up = lora_weights[key]
                                down = lora_weights[down_key]
                                if len(up.shape) < 2 or len(down.shape) < 2:
                                    continue
                                pair = {
                                    'base': base_key,
                                    'up': up,
                                    'down': down,
                                    'up_shape': up.shape,
                                    'down_shape': down.shape,
                                    'delta_shape': (up.shape[0], down.shape[1])
                                }
                                processed['all'].append(pair)
                                if 'te_' in key or 'text_model' in key:
                                    processed['text'].append(pair)
                                else:
                                    processed['unet'].append(pair)
                    self.loaded_loras[name] = processed
                    return processed
                except Exception as e:
                    print(f"Error loading LoRA: {e}")
        raise FileNotFoundError(f"LoRA file not found: {name}")

    def apply_loras(self, model, loras):
        if not loras:
            return None
        original_weights = {}
        applied_count = 0
        named_parameters = list(model.named_parameters())
        attention_layers = [(name, param) for name, param in named_parameters 
                           if ('attention' in name or 'attn' in name or 'proj' in name)
                           and name.endswith('.weight')]
        for name, scale in loras:
            lora_data = self.load_lora(name)
            for param_name, param in attention_layers:
                if len(param.shape) != 2:
                    continue
                out_dim, in_dim = param.shape
                for pair in lora_data['all']:
                    up, down = pair['up'], pair['down']
                    if (len(up.shape) == 2 and len(down.shape) == 2 and
                        up.shape[1] == down.shape[0] and
                        up.shape[0] == out_dim and down.shape[1] == in_dim):
                        try:
                            if param_name not in original_weights:
                                original_weights[param_name] = param.data.clone()
                            with torch.no_grad():
                                delta = scale * (up @ down)
                                param.data.add_(delta.to(param.device))
                                applied_count += 1
                                print(f"Applied LoRA to {param_name}")
                                break
                        except Exception as e:
                            print(f"Error applying LoRA to {param_name}: {e}")
        print(f"Applied {applied_count} LoRA weights")
        return original_weights if applied_count > 0 else None

    def restore_weights(self, model, original_weights):
        if not original_weights:
            return
        restored = 0
        with torch.no_grad():
            for name, original in original_weights.items():
                for param_name, param in model.named_parameters():
                    if param_name == name:
                        param.data.copy_(original)
                        restored += 1
                        break
        print(f"Restored {restored} weights")
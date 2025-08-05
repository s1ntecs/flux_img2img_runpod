import os
import torch

from diffusers import FluxControlPipeline
from image_gen_aux import DepthPreprocessor

# from huggingface_hub import hf_hub_download

# ------------------------- каталоги -------------------------
os.makedirs("loras", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ------------------------- пайплайн -------------------------
def get_pipeline():
    FluxControlPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Depth-dev",
        torch_dtype=torch.bfloat16,          # smaller weights
        device_map="cpu",                    # avoids cuda init
        low_cpu_mem_usage=True               # 🤗 transformers arg
    )
    DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")


if __name__ == "__main__":
    get_pipeline()

import os
import torch

from diffusers import FluxControlPipeline

# from huggingface_hub import hf_hub_download

# ------------------------- каталоги -------------------------
os.makedirs("loras", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ------------------------- пайплайн -------------------------
def get_pipeline():
    FluxControlPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Canny-dev",
        torch_dtype=torch.bfloat16
    )


if __name__ == "__main__":
    get_pipeline()

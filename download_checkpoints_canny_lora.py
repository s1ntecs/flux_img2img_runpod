import os
import torch

from controlnet_aux import CannyDetector
from diffusers import FluxControlPipeline

# from huggingface_hub import hf_hub_download

# ------------------------- каталоги -------------------------
os.makedirs("loras", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------- пайплайн -------------------------
def get_pipeline():
    pipe = FluxControlPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    )
    pipe.load_lora_weights(
        "black-forest-labs/FLUX.1-Canny-dev-lora"
    )


if __name__ == "__main__":
    get_pipeline()

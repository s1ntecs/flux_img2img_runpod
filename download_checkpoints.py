import os
import torch

from diffusers import FluxControlNetModel, FluxControlNetPipeline
from image_gen_aux import DepthPreprocessor

# from huggingface_hub import hf_hub_download

# ------------------------- каталоги -------------------------
os.makedirs("loras", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


# ------------------------- пайплайн -------------------------
def get_pipeline():
    base_repo = "black-forest-labs/FLUX.1-dev"
    controlnet_repo = "Shakker-Labs/FLUX.1-dev-ControlNet-Depth"

    CONTROLNET = FluxControlNetModel.from_pretrained(
        controlnet_repo, torch_dtype=torch.bfloat16
    )

    FluxControlNetPipeline.from_pretrained(
        base_repo,
        controlnet=CONTROLNET,
        torch_dtype=torch.bfloat16
    )

    DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")


if __name__ == "__main__":
    get_pipeline()

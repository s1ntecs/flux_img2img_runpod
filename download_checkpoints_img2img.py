import os
import torch

from diffusers import FluxControlNetImg2ImgPipeline, FluxControlNetModel
from image_gen_aux import DepthPreprocessor

# from huggingface_hub import hf_hub_download

# ------------------------- каталоги -------------------------
os.makedirs("loras", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ------------------------- пайплайн -------------------------
def get_pipeline():
<<<<<<< HEAD
    controlnet = FluxControlNetModel.from_pretrained(
        "InstantX/FLUX.1-dev-Controlnet-Canny",
        torch_dtype=torch.bfloat16
    )
    FluxControlNetImg2ImgPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=True,
        # map_location="cpu",
=======
    FluxControlPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Depth-dev",
        torch_dtype=torch.bfloat16,          # smaller weights
        device_map="cpu",                    # avoids cuda init
        low_cpu_mem_usage=True               # 🤗 transformers arg
>>>>>>> 60485d3e36e836b38aabb97471eae8b9cf6063c6
    )
    DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")


if __name__ == "__main__":
    get_pipeline()

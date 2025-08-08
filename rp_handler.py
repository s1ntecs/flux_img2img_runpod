# import cv2
import base64, io, random, time, numpy as np, torch
from typing import Any, Dict
from PIL import Image, ImageFilter

from diffusers import FluxControlNetPipeline, FluxControlNetModel
from image_gen_aux import DepthPreprocessor

import runpod
from runpod.serverless.utils.rp_download import file as rp_file
from runpod.serverless.modules.rp_logger import RunPodLogger

# --------------------------- КОНСТАНТЫ ----------------------------------- #
MAX_SEED = np.iinfo(np.int32).max
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MAX_STEPS = 250

TARGET_RES = 1024

logger = RunPodLogger()


# ------------------------- ФУНКЦИИ-ПОМОЩНИКИ ----------------------------- #
def filter_items(colors_list, items_list, items_to_remove):
    keep_c, keep_i = [], []
    for c, it in zip(colors_list, items_list):
        if it not in items_to_remove:
            keep_c.append(c)
            keep_i.append(it)
    return keep_c, keep_i


def resize_dimensions(dimensions, target_size):
    w, h = dimensions
    if w < target_size and h < target_size:
        return dimensions
    if w > h:
        ar = h / w
        return target_size, int(target_size * ar)
    ar = w / h
    return int(target_size * ar), target_size


def url_to_pil(url: str) -> Image.Image:
    info = rp_file(url)
    return Image.open(info["file_path"]).convert("RGB")


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def round_to_multiple(x, m=8):
    return (x // m) * m


def compute_work_resolution(w, h, max_side=1024):
    # масштабируем так, чтобы большая сторона <= max_side
    scale = min(max_side / max(w, h), 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # выравниваем до кратных 8
    new_w = round_to_multiple(new_w, 8)
    new_h = round_to_multiple(new_h, 8)
    return max(new_w, 8), max(new_h, 8)


# ------------------------- ЗАГРУЗКА МОДЕЛЕЙ ------------------------------ #
# БАЗА: FLUX.1-dev + depth ControlNet
base_repo = "black-forest-labs/FLUX.1-dev"
controlnet_repo = "Shakker-Labs/FLUX.1-dev-ControlNet-Depth"

CONTROLNET = FluxControlNetModel.from_pretrained(
    controlnet_repo, torch_dtype=DTYPE
)

PIPELINE = FluxControlNetPipeline.from_pretrained(
    base_repo,
    controlnet=CONTROLNET,
    torch_dtype=DTYPE
).to(DEVICE)

processor = DepthPreprocessor.from_pretrained(
    "LiheYoung/depth-anything-large-hf")


# ------------------------- ОСНОВНОЙ HANDLER ------------------------------ #
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = job.get("input", {})
        image_url = payload.get("image_url")
        if not image_url:
            return {"error": "'image_url' is required"}
        prompt = payload.get("prompt")
        if not prompt:
            return {"error": "'prompt' is required"}

        # ⚙️ Новые параметры
        neg = payload.get("negative_prompt") or \
              "blurry, low quality, watermark, logo, text, artifacts, distorted geometry, misaligned perspective"
        neg2 = payload.get("negative_prompt_2")  # опционально
        true_cfg_scale = float(payload.get("true_cfg_scale", 3.0))  # >1 чтобы включить «true CFG»
        cn_scale = float(payload.get("controlnet_conditioning_scale", 0.5))  # depth-рекомендация 0.3–0.7

        guidance_scale = float(payload.get("guidance_scale", 3.5))
        steps = min(int(payload.get("steps", MAX_STEPS)), MAX_STEPS)

        seed = int(payload.get("seed", random.randint(0, MAX_SEED)))
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        image_pil = url_to_pil(image_url)

        orig_w, orig_h = image_pil.size
        work_w, work_h = compute_work_resolution(orig_w, orig_h, TARGET_RES)
        image_pil = image_pil.resize((work_w, work_h), Image.Resampling.LANCZOS)

        # depth-карта
        control_image = processor(image_pil)[0].convert("RGB")

        # ------------------ генерация ---------------- #
        images = PIPELINE(
            prompt=prompt,
            prompt_2=None,  # или свой стиль для t5-второго энкодера
            negative_prompt=neg,               # ✅ теперь поддерживается
            negative_prompt_2=neg2,            # опционально
            true_cfg_scale=true_cfg_scale,     # ✅ включает негативку
            control_image=control_image,
            controlnet_conditioning_scale=cn_scale,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,     # для Flux обычно 3–5 норм
            generator=generator,
            width=work_w,
            height=work_h,
        ).images

        return {
            "images_base64": [pil_to_b64(i) for i in images],
            "time": round(time.time() - job["created"], 2) if "created" in job else None,
            "steps": steps,
            "seed": seed
        }

    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "CUDA out of memory" in str(exc):
            return {"error": "CUDA OOM — уменьшите 'steps' или размер изображения."}
        return {"error": str(exc)}
    except Exception as exc:
        import traceback
        return {"error": str(exc), "trace": traceback.format_exc(limit=5)}


# ------------------------- RUN WORKER ------------------------------------ #
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

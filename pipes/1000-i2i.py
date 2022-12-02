from oneflow import autocast
import requests
import oneflow as torch
from PIL import Image
from io import BytesIO
import os

from diffusers import (
    OneFlowStableDiffusionImg2ImgPipeline as StableDiffusionImg2ImgPipeline,
)

init_image = Image.open("sketch-mountains-input.jpg").convert("RGB")
init_image = init_image.resize((768, 512))

# load the pipeline
device = "cuda"
model_id_or_path = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id_or_path, revision="fp16", torch_dtype=torch.float16, use_auth_token=True
)


# or download via git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
# and pass `model_id_or_path="./stable-diffusion-v1-4"` without having to use `use_auth_token=True`.
pipe = pipe.to(device)

import os

os.environ["ONEFLOW_MLIR_ENABLE_TIMING"] = "1"
os.environ["ONEFLOW_MLIR_PRINT_STATS"] = "1"
os.environ["ONEFLOW_MLIR_CSE"] = "1"
os.environ["ONEFLOW_MLIR_GROUP_MATMUL"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"

prompt = "A fantasy landscape, trending on artstation"

output_dir = "oneflow-sd-i2i-output"
os.makedirs(output_dir, exist_ok=True)
with autocast("cuda"):
    for j in range(1000):
        images = pipe(
            prompt=prompt,
            init_image=init_image,
            strength=0.75,
            guidance_scale=7.5,
            compile_unet=True,
        ).images
        for i, image in enumerate(images):
            prompt = prompt.strip().replace("\n", " ")
            dst = os.path.join(output_dir, f"{prompt[:100]}-{j}-{i}.png")
            image.save(dst)

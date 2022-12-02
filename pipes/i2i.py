from oneflow import autocast
import requests
import oneflow as torch
from PIL import Image
from io import BytesIO

from diffusers import (
    OneFlowStableDiffusionImg2ImgPipeline as StableDiffusionImg2ImgPipeline,
)

init_image = Image.open("WechatIMG3270.jpeg").convert("RGB")
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

prompt = "A phone riding on a duck"

with autocast("cuda"):
    images = pipe(
        prompt=prompt,
        init_image=init_image,
        strength=0.75,
        guidance_scale=7.5,
        compile_unet=True,
    ).images

images[0].save("fantasy_landscape.png")

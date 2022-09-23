# CUDA_VISIBLE_DEVICES=1
# make sure you're logged in with `huggingface-cli login`
from torch import autocast
import torch
from diffusers import StableDiffusionPipeline
from diffusers import (
    FlaxStableDiffusionPipeline,
    FlaxDiffusionPipeline,
    StableDiffusionOnnxPipeline,
)

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
)
# pipe = FlaxDiffusionPipeline.from_pretrained(
#     "CompVis/stable-diffusion-v1-4", use_auth_token=True,
# )
# pipeline = FlaxStableDiffusionPipeline.from_pretrained(
#     "CompVis/stable-diffusion-v1-4", use_auth_token=True
# )
# pipeline = StableDiffusionOnnxPipeline.from_pretrained(
#     "CompVis/stable-diffusion-v1-4", use_auth_token=True
# )
pipe = pipe.to("cuda")

prompt = "we don't need no education"
with autocast("cuda"):
    images = pipe(prompt).images
    for i, image in enumerate(images):
        image.save(f"{prompt}-{i}.png")
import oneflow

print(oneflow.randn(2, 3))

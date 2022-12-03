# CUDA_VISIBLE_DEVICES=1
# make sure you're logged in with `huggingface-cli login`
from torch import autocast
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
)

pipe = pipe.to("cuda")
prompt = "a photo of an astronaut riding a horse on mars"
with autocast("cuda"):
    images = pipe(prompt).images
    images = pipe(prompt).images
    images = pipe(prompt).images
    images = pipe(prompt).images
    images = pipe(prompt).images
    for i in range(1000):
        images = pipe(prompt).images
        for i, image in enumerate(images):
            image.save(f"{prompt}-{i}.png")

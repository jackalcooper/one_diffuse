# CUDA_VISIBLE_DEVICES=1
# make sure you're logged in with `huggingface-cli login`
from torch import autocast
import torch
from diffusers import StableDiffusionPipeline
from diffusers import DDPMScheduler
ddpm = DDPMScheduler.from_config(
    "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
)
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
    scheduler=ddpm
)
import os
output_dir = "torch-sd-output"
os.makedirs(output_dir, exist_ok=True)
from timeit import default_timer as timer

pipe = pipe.to("cuda")
prompt = "a photo of an astronaut riding a horse on mars"

with autocast("cuda"):
    for i in range(1000):
        images = pipe(prompt).images
        for i, image in enumerate(images):
            image.save(f"{prompt}-{i}.png")
            dst = os.path.join(output_dir, f"{prompt[:100]}-{i}.png")
            image.save(dst)

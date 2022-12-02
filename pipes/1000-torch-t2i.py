# CUDA_VISIBLE_DEVICES=1
# make sure you're logged in with `huggingface-cli login`
from torch import autocast
import torch
import os
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
)

pipe = pipe.to("cuda")
prompt = "a photo of an astronaut riding a horse on mars"
prompt = """
a dog, baroque painting, beautiful detailed intricate insanely detailed octane render trending on artstation, 8 k artistic photography, photorealistic, soft natural volumetric cinematic perfect light, chiaroscuro, award - winning photograph
"""
output_dir = "torch-sd-output"
os.makedirs(output_dir, exist_ok=True)
with autocast("cuda"):
     for j in range(1000):
        images = pipe(prompt).images
        for i, image in enumerate(images):
            prompt = prompt.strip().replace("\n", " ")
            dst = os.path.join(output_dir, f"{prompt[:100]}-{j}-{i}.png")
            image.save(dst)

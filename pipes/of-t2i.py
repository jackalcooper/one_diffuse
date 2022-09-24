# CUDA_VISIBLE_DEVICES=1
# make sure you're logged in with `huggingface-cli login`
import torch
import oneflow
from torch import autocast
from diffusers import OneFlowStableDiffusionPipeline as StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
)

pipe = pipe.to("cuda")

prompt = "we don't need no education"
with autocast("cuda"):
    with oneflow.autocast("cuda"):
        images = pipe(prompt).images
        for i, image in enumerate(images):
            image.save(f"{prompt}-{i}.png")

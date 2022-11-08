import argparse
import oneflow as torch
from diffusers import OneFlowStableDiffusionPipeline

pipe = OneFlowStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
)

pipe = pipe.to("cuda")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple demo of image generation.")
    parser.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse on mars")
    args = parser.parse_args()
    return args

args = parse_args()
import os
output_dir = "oneflow-sd-output"
os.makedirs(output_dir, exist_ok=True)
with torch.autocast("cuda"):
    for j in range(100):
        images = pipe(args.prompt).images
        for i, image in enumerate(images):
            dst = os.path.join(output_dir, f"{args.prompt[:100]}-{i}-{j}.png")
            image.save(dst)

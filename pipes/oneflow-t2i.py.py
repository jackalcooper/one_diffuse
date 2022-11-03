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
with torch.autocast("cuda"):
    images = pipe(args.prompt).images
    for i, image in enumerate(images):
        image.save(f"{args.prompt[:20]}-{i}.png")

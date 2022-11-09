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
from timeit import default_timer as timer
with torch.autocast("cuda"):
    for j in range(100):
        prompt = args.prompt
        prompt = """
        But when the melancholy fit shall fall
Sudden from heaven like a weeping cloud,
That fosters the droop-headed flowers all,
And hides the green hill in an April shroud;
Then glut thy sorrow on a morning rose,
Or on the rainbow of the salt sand-wave,
Or on the wealth of globed peonies …
        """
        start = timer()
        images = pipe(prompt, compile_unet=True).images
        print("[oneflow]", "[elapsed(s)]", "[pipe]", f"{timer() - start}")
        save_start = timer()
        for i, image in enumerate(images):
            dst = os.path.join(output_dir, f"{prompt.strip()[:100]}-{j}-{i}.png")
            image.save(dst)
        print("[oneflow]", "[elapsed(s)]", "[save]", f"{timer() - save_start}")

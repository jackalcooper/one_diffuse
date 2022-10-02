# CUDA_VISIBLE_DEVICES=1
# make sure you're logged in with `huggingface-cli login`
import oneflow as torch
from diffusers import OneFlowStableDiffusionPipeline, OneFlowUNet2DConditionModel

pipe = OneFlowStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
)

pipe = pipe.to("cuda")

prompt = "apple made a toaster that makes toast shaped like an apple"

with torch.autocast("cuda"):
    images = pipe(prompt, num_inference_steps=4).images
    for i, image in enumerate(images):
        image.save(f"{prompt}-{i}.png")

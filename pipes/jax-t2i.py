# CUDA_VISIBLE_DEVICES=1
# make sure you're logged in with `huggingface-cli login`
from diffusers import FlaxStableDiffusionPipeline

pipe = FlaxStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", use_auth_token=True
)

prompt = "we don't need no education"

images = pipe(prompt).images
for i, image in enumerate(images):
    image.save(f"{prompt}-{i}.png")

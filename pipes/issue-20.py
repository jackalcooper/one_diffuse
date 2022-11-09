import oneflow as torch
from diffusers import OneFlowStableDiffusionPipeline
import timeit

pipe = OneFlowStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
)

pipe = pipe.to("cuda")

start = timeit.default_timer()
prompt = "a photo of an astronaut riding a horse on mars"
with torch.autocast("cuda"):
    images = pipe(prompt).images
    for i, image in enumerate(images):
        image.save(f"{prompt}-of-{i}.png")

end = timeit.default_timer()
print('Running time: %s Seconds' % (end - start))

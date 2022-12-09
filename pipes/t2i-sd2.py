import oneflow as torch
from diffusers import (
    OneFlowStableDiffusionPipeline as StableDiffusionPipeline,
    OneFlowEulerDiscreteScheduler as EulerDiscreteScheduler,
)

model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
import os

output_dir = "oneflow-sd-output"
os.makedirs(output_dir, exist_ok=True)
os.environ["ONEFLOW_MLIR_ENABLE_TIMING"] = "1"
os.environ["ONEFLOW_MLIR_PRINT_STATS"] = "1"
os.environ["ONEFLOW_MLIR_CSE"] = "1"
os.environ["ONEFLOW_MLIR_GROUP_MATMUL"] = "1"
os.environ["ONEFLOW_MLIR_CSE"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
os.environ["ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
os.environ["ONEFLOW_KERENL_ENABLE_TRT_FLASH_ATTN_IMPL"] = "1"

with torch.autocast("cuda"):
    for j in range(1000):
        pipe.set_unet_graphs_cache_size(8)
        width = random.choice([128, 256, 512, 768])
        height = random.choice([128, 256, 512, 768])
        width =768+128*random.choice([0,1,2])
        height =768+128*random.choice([0,1,2])
        images = pipe(prompt, height=height, width=width).images
        # images = pipe(prompt, height=768, width=768).images
        # images = pipe(prompt, height=512, width=512).images
        for i, image in enumerate(images):
            prompt = prompt.strip().replace("\n", " ")
            dst = os.path.join(output_dir, f"{prompt[:100]}-{j}-{i}.png")
            image.save(dst)

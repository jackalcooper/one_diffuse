import os

os.environ["ONEFLOW_MLIR_CSE"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
os.environ["ONEFLOW_MLIR_GROUP_MATMUL"] = "1"
os.environ["ONEFLOW_MLIR_PREFER_NHWC"] = "1"

os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_CONV_BIAS"] = "1"
os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR"] = "1"

os.environ["ONEFLOW_KERENL_CONV_ENABLE_CUTLASS_IMPL"] = "1"
os.environ["ONEFLOW_KERENL_FMHA_ENABLE_TRT_FLASH_ATTN_IMPL"] = "1"
os.environ["ONEFLOW_KERNEL_GLU_ENABLE_DUAL_GEMM_IMPL"] = "1"

os.environ["ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
os.environ["ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
os.environ["ONEFLOW_KERENL_CONV_CUTLASS_IMPL_ENABLE_TUNING_WARMUP"] = "1"

import oneflow as torch
from diffusers import (
    OneFlowStableDiffusionPipeline as StableDiffusionPipeline,
    OneFlowEulerDiscreteScheduler as EulerDiscreteScheduler,
)
import random
import diffusers

diffusers.logging.set_verbosity_info()
model_id = "stabilityai/stable-diffusion-2"
model_id = "stabilityai/stable-diffusion-2-1"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"


output_dir = "oneflow-sd-output"
os.makedirs(output_dir, exist_ok=True)

os.environ["ONEFLOW_MLIR_PRINT_STATS"] = "1"
os.environ["ONEFLOW_NNGRAPH_ENABLE_PROGRESS_BAR"] = "1"
pipe.set_unet_graphs_cache_size(8)
with torch.autocast("cuda"):
    for j in range(1000):
        width = random.choice([128, 256, 512, 768])
        height = random.choice([128, 256, 512, 768])
        width = 768 + 128 * random.choice([0, 1, 2])
        height = 768 + 128 * random.choice([0, 1, 2])
        height = 768
        width = 768
        images = pipe(prompt, height=height, width=width).images
        # images = pipe(prompt, height=768, width=768).images
        # images = pipe(prompt, height=512, width=512).images
        for i, image in enumerate(images):
            prompt = prompt.strip().replace("\n", " ")
            sanitized = model_id.replace("/", "-")
            dst = os.path.join(output_dir, f"{prompt[:100]}-{j}-{i}-{sanitized}.png")
            image.save(dst)

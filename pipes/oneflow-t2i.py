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


os.environ["ONEFLOW_MLIR_PRINT_STATS"] = "1"
os.environ["ONEFLOW_NNGRAPH_ENABLE_PROGRESS_BAR"] = "1"

# CUDA graph
os.environ["ONEFLOW_MLIR_FUSE_KERNEL_LAUNCH"] = "1"
os.environ["ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH"] = "1"

import argparse
import oneflow as torch
from diffusers import OneFlowStableDiffusionPipeline
import random
from diffusers import OneFlowDPMSolverMultistepScheduler as DPMSolverMultistepScheduler
from diffusers import OneFlowDDPMScheduler as DDPMScheduler


dpm_solver = DPMSolverMultistepScheduler.from_config(
    "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
)
ddpm = DDPMScheduler.from_config("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

model = "CompVis/stable-diffusion-v1-4"
pipe = OneFlowStableDiffusionPipeline.from_pretrained(
    model,
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
    # scheduler=dpm_solver,
    # scheduler=ddpm,
    safety_checker=None,
)

pipe = pipe.to("cuda")


def dummy(images, **kwargs):
    return images, False


# pipe.safety_checker = dummy


def parse_args():
    parser = argparse.ArgumentParser(description="Simple demo of image generation.")
    parser.add_argument(
        "--prompt", type=str, default="a photo of an astronaut riding a horse on mars"
    )
    args = parser.parse_args()
    return args


args = parse_args()

output_dir = "oneflow-sd-output"
os.makedirs(output_dir, exist_ok=True)
from timeit import default_timer as timer
import diffusers

diffusers.logging.set_verbosity_info()
with torch.autocast("cuda"):
    for j in range(1000):
        prompt = args.prompt
        # prompt = """
        # a dog, baroque painting, beautiful detailed intricate insanely detailed octane render trending on artstation, 8 k artistic photography, photorealistic, soft natural volumetric cinematic perfect light, chiaroscuro, award - winning photograph
        # """
        start = timer()
        pipe.set_unet_graphs_cache_size(8)
        width = 512
        height = 512
        # width = random.choice([128, 256, 512, 768])
        # height = random.choice([128, 256, 512, 768])
        # width =768+128*random.choice([0,1,2])
        # height =768+128*random.choice([0,1,2])
        num_images_per_prompt = 1
        # num_images_per_prompt = random.choice([1, 2, 3, 4])
        num_inference_steps = 50
        if isinstance(pipe.scheduler, DPMSolverMultistepScheduler):
            num_inference_steps = 20
        images = pipe(
            prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            compile_unet=True,
            # compile_unet=False,
            # compile_vae=False,
            # unrolled_timesteps=True,
        ).images
        print(
            "[oneflow]",
            f"[{width}x{height}]",
            "[elapsed(s)]",
            "[pipe]",
            f"{timer() - start}",
        )
        save_start = timer()
        for i, image in enumerate(images):
            prompt = prompt.strip().replace("\n", " ")
            dst = os.path.join(output_dir, f"{prompt[:100]}-{j}-{i}.png")
            image.save(dst)
        print("[oneflow]", "[elapsed(s)]", "[save]", f"{timer() - save_start}")

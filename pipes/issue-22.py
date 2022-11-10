from diffusers import StableDiffusionPipeline, OneFlowStableDiffusionPipeline
import oneflow as torch

model_path = "CompVis/stable-diffusion-v1-4"
prompt = "a cute girl, blue eyes, brown hair"

pipe = OneFlowStableDiffusionPipeline.from_pretrained(
        model_path,
        # scheduler=DDIMScheduler(
        #     beta_start=0.00085,
        #     beta_end=0.012,
        #     beta_schedule="scaled_linear",
        #     clip_sample=False,
        #     set_alpha_to_one=True,
        # )
    )

def dummy(images, **kwargs):
    return images, False
pipe.safety_checker = dummy
pipe = pipe.to("cuda")
image = pipe(prompt, num_inference_steps=30).images[0]
image.save(f"output.png")

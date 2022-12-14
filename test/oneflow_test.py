import oneflow as torch
from diffusers import (
    OneFlowDPMSolverMultistepScheduler as DPMSolverMultistepScheduler,
    OneFlowStableDiffusionPipeline as StableDiffusionPipeline,
)

# model_id = "./weights/stable-diffusion"
model_id = "CompVis/stable-diffusion-v1-4"
dpm_solver = DPMSolverMultistepScheduler.from_config(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, scheduler=dpm_solver, revision="fp16",             safety_checker=None
)
pipe.set_unet_graphs_cache_size(8)
pipe = pipe.to("cuda")
from timeit import default_timer as timer
def run():
    with torch.autocast("cuda"):
        args = [
            {
                "count": 1,
                "face_enhance": False,
                "guid_scale": 7.5,
                "height": 512,
                "init_image": "",
                "model": "anime",
                "negative_prompt": "",
                "prompt": "(a country mouse and a city mouse )",
                "safety_checker": True,
                "seed": 0,
                "steps": 15,
                "strength": 0.5,
                "upscale_factor": 2,
                "version": "1.0",
                "width": 512,
            },
            {
                "count": 1,
                "face_enhance": True,
                "guid_scale": 7.5,
                "height": 400,
                "init_image": "",
                "model": "anime",
                "negative_prompt": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts,signature, watermark, username, blurry, artist name,bad feet,cropped,bad hands, missing arms, long neck, Humpbacked,large breasts,Low facial detail",
                "prompt": "(elf, cum, thick thighs, stomach, cum on body, cum on face, blush, green hair,  green eyes, torn clothes, full body, cum puddle, tears, cum bath, laying down, tied up, spider web, spider eggs, wide hips, ass focus, curvey, defeated \n\n\n), ((masterpiece)),\xa0(((best\xa0quality))),\xa0((ultra-detailed)),\xa0((illustration)),\xa0((disheveled\xa0hair))",
                "safety_checker": False,
                "seed": 0,
                "steps": 15,
                "strength": 0.5,
                "upscale_factor": 4,
                "version": "1.0",
                "width": 704,
            },
        ]
        for arg in args:
            images = pipe(
                arg["prompt"],
                negative_prompt=arg["negative_prompt"],
                width=arg["width"],
                height=arg["height"],
                num_inference_steps=15,
            ).images
            print(len(images))
            # images[0].save(f"i.png")

for i in range(1000):
    start = timer()
    run()
    print("[ela]", timer() - start)

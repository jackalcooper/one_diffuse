import torch

# import oneflow as torch
import oneflow

print("torch version:", torch.__version__)
from diffusers import AutoencoderKL

# from diffusers import OneFlowAutoencoderKL as AutoencoderKL
import oneflow as flow

print("oneflow version:", flow.__version__)

vae = (
    AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="vae",
        torch_dtype=torch.float16,
        revision="fp16",
        use_auth_token=True,
    )
    .to(torch.device("cuda"))
    .eval()
    .requires_grad_(False)
)

randoma = torch.randn(1, 4, 96, 96).cuda().half()
testresult = vae.decode(randoma)["sample"]
print(testresult)

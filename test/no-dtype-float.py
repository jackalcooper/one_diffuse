def f(m):
    torch = m
    print(f"{torch.__path__=}")
    fp32_scalar_t = torch.tensor(14.6146)
    fp16_t = torch.tensor([14.6146], dtype=torch.float16)
    fp16_scalar_t = torch.tensor(14.6146, dtype=torch.float16)
    float_py = 14.6146

    print(f"{(fp16_t * fp32_scalar_t).dtype=}")
    print(f"{(fp32_scalar_t * fp16_t).dtype=}")
    print(f"{(fp16_scalar_t * fp32_scalar_t).dtype=}")
    print(f"{(fp32_scalar_t * fp16_scalar_t).dtype=}")
    print(f"{(fp16_scalar_t * float_py).dtype=}")
    print(f"{(float_py * fp16_scalar_t).dtype=}")


import torch

f(torch)
import oneflow

f(oneflow)

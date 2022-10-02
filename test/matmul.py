import argparse

parser = argparse.Arparser = argparse.ArgumentParser(
    description="Process some integers."
)
parser.add_argument(
    "--pt", default=False, action="store_true", help="If specified, use pytorch",
)
args = parser.parse_args()
if args.pt:
    import torch
else:
    import oneflow as torch
print("torch __path__:", torch.__path__)
dtype = torch.float32

x = torch.randn(10, 512, 512, dtype=dtype).to("cuda")
b = torch.randn(10, 512, 512, dtype=dtype).to("cuda")
w = torch.randn(10, 512, 512, dtype=dtype).to("cuda")
for i in range(10000):
    original = x
    y = torch.matmul(x, w) + b
    x = y + original
print(x.dtype)
print(x.device)

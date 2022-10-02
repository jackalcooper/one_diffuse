import argparse
from typing import Optional, Tuple
import numpy as np

parser = argparse.Arparser = argparse.ArgumentParser(
    description="Process some integers."
)
parser.add_argument(
    "--pt", default=False, action="store_true", help="If specified, use pytorch",
)
args = parser.parse_args()
if args.pt:
    import torch
    from torch import Tensor
    import torch.nn as nn
    import torch.nn.functional as F
else:
    import oneflow as torch
    from oneflow import Tensor
    import oneflow.nn as nn
    import oneflow.nn.functional as F

print("torch __path__:", torch.__path__)
dtype = torch.float32


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float("Inf"))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


scaled_dot_attn = ScaledDotProductAttention(12)

key = torch.randn(4, 3, 2).to("cuda").float()
value = torch.randn(4, 3, 2).to("cuda").float()
query = torch.randn(4, 3, 2).to("cuda").float()
for i in range(100000):
    context, attn = scaled_dot_attn(query, key, value)
    query, key, value = context, context, context

print(context.dtype)
print(context.device)

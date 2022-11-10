# import torch # ok
import oneflow as torch # not ok


scalar = torch.tensor(741, dtype=torch.int64)
t = torch.tensor([scalar] * 3, dtype=torch.long, device="cpu")

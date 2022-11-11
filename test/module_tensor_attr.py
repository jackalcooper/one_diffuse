import oneflow as torch
import oneflow.nn as nn

class SomeModule(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.x = torch.randn(5, 5)

    def forward(
        self,
    ):
        index = torch.tensor([1, 1])
        return self.x[index]

class SomeGraph(torch.nn.Graph):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def build(self):
        return self.m()

m = SomeModule()
m()
g = SomeGraph(m)
g()

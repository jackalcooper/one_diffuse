import oneflow as torch
import oneflow.nn as nn

class SomeModule(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        linear1 = nn.Linear(5, 5, bias=False)
        linear2 = nn.Linear(5, 5, bias=False)
        linear3 = nn.Linear(5, 5, bias=False)
        self.modulelist = nn.ModuleList([linear1, linear2, linear3])

    def forward(
        self,
        x
    ):
        y = self.modulelist[0](torch.randn(5, 5))
        print(type(self), f"{len(self.modulelist)=}")
        print(type(self), f"{len(self.modulelist[1:])=}")
        return y

class SomeGraph(torch.nn.Graph):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def build(self, x):
        return self.m(x)

m = SomeModule()
m(torch.randn(5, 5))
g = SomeGraph(m)
g(torch.randn(5, 5))

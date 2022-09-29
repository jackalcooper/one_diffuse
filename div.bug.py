import torch
import oneflow


def dummy_sample_deter(mod):
    num_elems = 20
    sample = mod.arange(num_elems)
    sample = sample / num_elems

    print(mod.__name__, sample.dtype)


dummy_sample_deter(oneflow)
dummy_sample_deter(torch)

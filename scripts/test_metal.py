from d3sim.csrc import build
import torch 
from cumm import tensorview as tv
from d3sim import d3sim_cc_torch
from d3sim.d3sim_cc_torch.d3sim_cc_torch import PyTorchTools
import numpy as np 
print(dir(PyTorchTools))

def main():

    aa = torch.rand(10).to("mps")
    a = aa[5:]
    print(a)

    a_tv = PyTorchTools.torch2tensor(a)

    print(a_tv.cpu().numpy())
    # b_np = np.random.uniform(-1, 1, size=[10, 10]).astype(np.float32)
    # b_tv = tv.from_numpy(b_np).cuda()
    # b_th = PyTorchTools.tensor2torch(b_tv, clone=False)

    # print(b_np, b_th)
    pass 

if __name__ == "__main__":
    main()
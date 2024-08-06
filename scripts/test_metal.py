from d3sim.csrc import build
import torch 
from cumm import tensorview as tv
import numpy as np 
from d3sim.csrc.inliner import INLINER

def main():

    aa = torch.rand(10).to("mps")
    INLINER.kernel_1d("test", aa.shape[0], 0, f"""
    $aa[i] = 1;
    """)

if __name__ == "__main__":
    main()
from cumm import tensorview as tv
import torch 
from cumm.common import TensorViewArrayLinalg, TensorViewNVRTC, TensorViewNVRTCHashKernel
from cumm.inliner import NVRTCInlineBuilder, MPSContextBase
from .rotation import RotationMath
from d3sim.d3sim_thtools.pytorch_tools import PyTorchTools
class MPSContext(MPSContextBase):
    def get_command_buffer(self) -> int:
        return PyTorchTools.mps_get_default_command_buffer()

    def get_dispatch_queue(self) -> int:
        return PyTorchTools.mps_get_default_dispatch_queue()

    def commit(self):
        return PyTorchTools.mps_commit()

INLINER = NVRTCInlineBuilder([
    TensorViewArrayLinalg, TensorViewNVRTC, TensorViewNVRTCHashKernel,
    RotationMath
],
                             reload_when_code_change=True, mps_context=MPSContext())

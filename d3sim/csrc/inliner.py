from cumm import tensorview as tv
import torch 
from cumm.common import TensorViewArrayLinalg, TensorViewNVRTC, TensorViewNVRTCHashKernel
from cumm.inliner import NVRTCInlineBuilder, MPSContextBase
from .rotation import RotationMath
from .camera import CameraOps, CameraDefs
from .vis import VisUtils
from .gs3d import Gaussian3D
import pccm
from d3sim.d3sim_thtools.pytorch_tools import PyTorchTools

class MPSContext(MPSContextBase):
    def get_command_buffer(self) -> int:
        return PyTorchTools.mps_get_default_command_buffer()

    def get_dispatch_queue(self) -> int:
        return PyTorchTools.mps_get_default_dispatch_queue()

    def commit(self):
        return PyTorchTools.mps_commit()

    def flush_command_encoder(self):
        return PyTorchTools.mps_flush_command_encoder()

__deps: list[type[pccm.Class]] = [
    TensorViewArrayLinalg, TensorViewNVRTC, TensorViewNVRTCHashKernel,
    RotationMath, CameraOps, CameraDefs, VisUtils, Gaussian3D
]

INLINER = NVRTCInlineBuilder(__deps,
                             reload_when_code_change=True, mps_context=MPSContext())
INLINER.maximum_1d_threads = 256
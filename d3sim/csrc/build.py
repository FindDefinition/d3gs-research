import pccm
from pathlib import Path
from ccimport import compat
from typing import List 

from pccm.utils import project_is_editable, project_is_installed
import torch 
from cumm.bindings import build_pytorch_bindings
from d3sim.constants import PACKAGE_ROOT
# torch_version = torch.__version__.split("+")[0]

build_pytorch_bindings("d3sim_thtools", PACKAGE_ROOT)

# torch_version = torch_version.replace(".", "_")
# __pytorch_cu = PyTorchTools()
# __pytorch_cu.namespace = "d3sim_cc_torch"
# pccm.builder.build_pybind([__pytorch_cu],
#                         PACKAGE_ROOT / "d3sim_cc_torch",
#                         build_dir=PACKAGE_ROOT / "build" / f"d3sim_cc_torch_{torch_version}",
#                         namespace_root=PACKAGE_ROOT / "csrc",
#                         verbose=True,
#                         load_library=False,
#                         std="c++17")



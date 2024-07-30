from tensorpc.flow import plus, mui
from tensorpc.flow.components.flowplus import ComputeFlow
from tensorpc.flow import mark_create_layout
from tensorpc.flow import appctx
import sys 
from tensorpc import PACKAGE_ROOT
import numpy as np 
from d3sim.data.datasets.waymo import loader

class ComputeFlowApp:
    @mark_create_layout
    def my_layout(self):
        appctx.get_app().set_enable_language_server(True)
        pyright_setting = appctx.get_app().get_language_server_settings()
        pyright_setting.python.analysis.pythonPath = sys.executable
        pyright_setting.python.analysis.extraPaths = [
            str(PACKAGE_ROOT.parent),
        ]
        self.cflow = ComputeFlow("d3sim_dev")
        self.panel = plus.InspectPanel({
            "a": np.zeros((100, 3)),
            "cflow_dev": ComputeFlow("d3sim_dev_another"),
        }, use_fast_tree=True, init_layout=self.cflow)
        return self.panel.prop(width="100%", height="100%", overflow="hidden")


from d3sim.constants import D3SIM_DEV_SECRETS_PATH
from d3sim.core import dataclass_dispatch as dataclasses 

import yaml
@dataclasses.dataclass
class SecretConstants:
    origin_3dgs_model_path: str 
    origin_3dgs_garden_dataset_path: str
    origin_3dgs_grad_path: str
    h3dgs_example_dataset_path: str


def load_secret_constants():
    path = D3SIM_DEV_SECRETS_PATH
    with open(path, "r") as f:
        secrets = yaml.safe_load(f)
    return SecretConstants(**secrets)


from pathlib import Path
import platform 

PACKAGE_ROOT = Path(__file__).parent.resolve()


D3SIM_FOLDER_CREATE_MODE = 0o755

InMacOS = False
if platform.system() == "Darwin":
    InMacOS = True

IsAppleSiliconMacOs = InMacOS and platform.machine() == "arm64"

D3SIM_DEFAULT_DEVICE = "cuda"
if IsAppleSiliconMacOs:
    D3SIM_DEFAULT_DEVICE = "mps"
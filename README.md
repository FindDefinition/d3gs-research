# d3sim

A personal project for 3dgs research.

## Run 

```bash
pip install -U cumm-cu114 # or pip install -U cumm-cu121

pip install -e .
```

then prepare original dataset, change path in `d3sim/algos/d3gs/train.py`, and run that file.
```
D3SIM_DISABLE_ARRAY_CHECK=1 python d3sim/algos/d3gs/train.py
```

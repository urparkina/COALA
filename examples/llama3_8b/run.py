import subprocess
import os
from typing import Any
from tempfile import NamedTemporaryFile
import yaml
import sys
from math import floor

from lib.Runner import run_experiment
from printer import beaty
    

with open(f'./examples/llama3_8b/config.yaml') as f:
    config = yaml.safe_load(f)


env = dict(
    BASE_MODEL="meta-llama/Llama-3.1-8B",
    MODEL_DIR=f"./test_/llama3_8b/",
    TMP_DIR="./test_/tmp",
    USE_FP16=os.environ.get("USE_FP16", "False"),
    CONTEXT_LENGTH="4096",
)


print(config)
code, results = run_experiment(env, config)
beaty(results[0])
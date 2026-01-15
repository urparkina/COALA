import yaml
from math import floor
import os
import sys

from lib.Runner import run_experiment
from examples.printer import beaty



with open(f'./examples/llama3_1b/config.yaml') as f:
    config = yaml.safe_load(f)


env = dict(
    # BASE_MODEL="meta-llama/Llama-3.2-1B",
    BASE_MODEL="meta-llama/Llama-3.2-1B-Instruct",
    MODEL_DIR=f"./test_/llama3_1b/",
    TMP_DIR="./test_/tmp",
    USE_FP16=os.environ.get("USE_FP16", "False"),
    CONTEXT_LENGTH="1024",
)

print(config)
code, results = run_experiment(env, config)

beaty(results[0])



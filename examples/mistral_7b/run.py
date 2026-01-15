import os
import yaml

from lib.Runner import run_experiment
from printer import beaty
    
    


with open(f'./examples/mistral_7b/config.yaml') as f:
    config = yaml.safe_load(f)



env = dict(
    BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2",
    # BASE_MODEL="mistralai/Mistral-7B-v0.1",
    MODEL_DIR=f"./test_/mistral_7b",
    TMP_DIR="./test_/tmp",
    USE_FP16=os.environ.get("USE_FP16", "False"),
    CONTEXT_LENGTH="4096",
)

print(config)
code, results = run_experiment(env, config)
beaty(results[0])

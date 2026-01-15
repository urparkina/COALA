import subprocess
import os
from typing import Any
from tempfile import NamedTemporaryFile
import yaml
import sys


def run_experiment(env, config, task='common-reasoning') -> tuple[int, list[Any]]:
    path_to_run = './scripts/run.sh'
    if task == 'math':
        path_to_run = './scripts/run_math.sh'
    elif task == 'text':
        path_to_run = './scripts/run_text.sh'
    elif task =='wiki':
        path_to_run = './scripts/run_wiki.sh'
    
    env = {**env, **os.environ}
    with NamedTemporaryFile("w") as config_file:
        cfg_string = yaml.dump(config)
        config_file.write(cfg_string)
        config_file.flush()

        env["CONFIG_PATH"] = config_file.name 
        status = subprocess.run(
            [path_to_run],
            shell=False,
            check=False,
            env=env,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
        )
    exit_code = status.returncode
    if exit_code != 0:
        return exit_code, []

    results = []
    string = status.stdout.decode()
    for line in string.split('\n'):
        line = line.strip()
        PREFIX = "results="
        if line.startswith(PREFIX):
            results.append(eval(line[len(PREFIX):]))
    results.append(string)
    return 0, results

#!/bin/bash
set -e
set -o pipefail 

export USE_FP16=True

python3 -m examples.llama3_1b.run


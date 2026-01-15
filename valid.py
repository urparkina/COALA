from dataclasses import dataclass, field
from typing import Optional
from lib import Models, Datasets
import transformers
from transformers import Trainer
import numpy as np
from transformers import Trainer
from omegaconf import OmegaConf
import lm_eval
from datasets import load_dataset
from lm_eval.models.huggingface import HFLM
from typing import List, Optional
import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"



@dataclass
class InferenceArguments(transformers.TrainingArguments):
    path: Optional[str] = field(default=None)
    
    name: Optional[str] = field(default="facebook/opt-125m")
            
    seed_of_gen: int = field(default=42, metadata={"help": "seed for generation dataset"})
    
    count_examples: int = field(default=None, metadata={"help": "count of examples will be load from dataset"})
        
    start_token: int = field(default=None, metadata={"help":"Token after which the model should learn to generate"})
    
    model_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)
    
    name_datasets: List[str] = field(
        default_factory=lambda: ["common-reasoning"],
        metadata={
            "help": (
                "Список датасетов."
            ),
            "nargs": "+",
        },
    )    
    config_path: str = field(default=None)

    
    
def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    predictions = eval_pred.predictions[0].reshape(labels.shape)
    sm = dict()
    sm['accuracy'] = 0
    k = labels.shape[0]
    
    for i in range(len(predictions)):
        mask = labels[i] != -100
        valid_labels = labels[i][mask]
        valid_predictions = np.roll(predictions[i], 1)[mask]
            
        all_correct = all(p == r for p, r in zip(valid_predictions, valid_labels))

        sm['accuracy'] += (1 if all_correct else 0) / k
        
    return sm


def valid():
    parser = transformers.HfArgumentParser(InferenceArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    print(script_args.name)
    
    if script_args.path == None:
        model, tokenizer = Models.LoadLLM(script_args.name)
    else:
        adaptor_config = OmegaConf.load(script_args.config_path)
        model, tokenizer = Models.LoadFineTuneLLM(adaptor_config, script_args.name, script_args.path)
    
    batch_size = script_args.per_device_eval_batch_size
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    if os.environ.get("USE_FP16", "False") == "True":
        model.half()
        
    task_names = script_args.name_datasets

    results = lm_eval.simple_evaluate(
        hflm,
        tasks=task_names,
        num_fewshot=0,
        batch_size=batch_size,
    )["results"]
    
    print(f'{results=}')


if __name__ == "__main__":
    valid()

from dataclasses import dataclass
from typing import List
from lib import Datasets
from transformers import Trainer
import torch
import tempfile
import os
from tqdm import tqdm
from datasets import load_dataset
from coala.Injector import inject_coala, prepare_get_samples, after_get_samples

METHODS_WITHOUT_SAMPLES = ['empty', 'svd']
METHODS_WITH_SAMPLES = ['svd_llm', 'svd_llm_2', 'coala', 'asvd']

@dataclass
class COALA_Config:
    ratio: int
    params: dict
    target_modules: List[str]
    compress_strategy: str = None
    samples: str = None
    fp16: bool = False
    adaptive_rank: bool = False
    
    
def compress_model(model, config, args=None, tokenizer=None, logs=None, ranks=None):
    if config.compress_strategy not in METHODS_WITHOUT_SAMPLES:
        model, hooks = prepare_get_samples(model, config)
        data = Datasets.LoadDataset('train', None, config.samples)
    
        dataset, data_collator = Datasets.PrepareDataset(**data, args=args, tokenizer=tokenizer, desc="Load dataset for compress")

        direct = args.output_dir
        
        with tempfile.TemporaryDirectory() as temp_dir:
            args.output_dir = temp_dir
            args.logging_dir = temp_dir

            trainer = Trainer(
                model=model,
                args=args,
                data_collator=data_collator,
                eval_dataset=dataset,
                tokenizer=tokenizer,
                preprocess_logits_for_metrics=Datasets.preprocess_logits_for_metrics
            )

            trainer.evaluate()
    
            after_get_samples(model, config, hooks)
    
        args.output_dir = direct
    
    return inject_coala(config, model, logs, ranks)

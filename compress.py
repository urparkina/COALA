import copy
import os
from dataclasses import dataclass, field
from typing import Optional
from lib import Models, Datasets
import torch
import transformers
from omegaconf import OmegaConf
from transformers import Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback, TrainerState, TrainerControl
from torch.utils.tensorboard import SummaryWriter
import pickle


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default=None)
    
    seed_of_gen: int = field(default=42, metadata={"help": "seed for generation dataset"})
            
    start_token: int = field(default=10000, metadata={"help":"Token after which the model should learn to generate"})
    
    model_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)
    
    config_path: str = field(default=None)

def compress():
    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    print(script_args.model_name_or_path)
    
    model, tokenizer = Models.LoadLLM(script_args.model_name_or_path)
    adaptor_config = OmegaConf.load(script_args.config_path)
    
    ranks, model = Models.PrepareModel(model, adaptor_config, script_args, tokenizer)
    # model.half()
        
    model.save_pretrained(script_args.output_dir, max_shard_size='1GB')
    tokenizer.save_pretrained(script_args.output_dir)
    with open(script_args.output_dir + '/ranks.pkl', 'wb') as f:
        pickle.dump(ranks, f)


if __name__ == "__main__":
    compress()
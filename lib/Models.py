from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_utils import load_sharded_checkpoint
from omegaconf import OmegaConf
from dataclasses import dataclass, fields
from coala.Config import COALA_Config
from coala.Config import compress_model
import pickle
import sys

def LoadLLM(name, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(name, device_map=device)

    return model, tokenizer


def LoadFineTuneLLM(config, name, path):
    with open(path + '/ranks.pkl', 'rb') as f:
        ranks = pickle.load(f)
    print(path, ranks, file=sys.stderr)
    model, _ = LoadLLM(name)
    tokenizer = AutoTokenizer.from_pretrained(path)
    _, model = PrepareModel(model, config, None, tokenizer, True, ranks)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    _ = load_sharded_checkpoint(
            model=model,
            folder=path,
            strict=False
        )
    
    return model, tokenizer

def PrepareModel(model, config_base, args, tokenizer, only_init=False, ranks=None):
    config_dict = {f.name: config_base.get(f.name) for f in fields(COALA_Config)}
    config = COALA_Config(**config_dict)
    print('Configuration', config)
    
    if only_init:
        config.compress_strategy = "empty"
    
    return compress_model(model, config, args, tokenizer, None, ranks)
    


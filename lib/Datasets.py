from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
import torch


def _tokenize_(example, name_text, tokenizer, max_length):
    
    t =  tokenizer(
        example[name_text],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    return t
    
class DataCollatorForChat(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False, mlm_probability=0.15, start_token=-1):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
        self.start_token = start_token

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        
        return batch

def LoadDataset(type, count, name="common-reasoning", seed=42):
    dataset = load_dataset(f"./datasets/{name}", split=type)
    
    dataset = dataset.shuffle(seed=seed)
    if count != None:
        dataset = dataset.select(range(count))
        
    return {"dataset": dataset, "name_text": "text"}


def PrepareDataset(dataset, name_text, args, tokenizer, desc=""):
    dataset = dataset.map(
        _tokenize_,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Running tokenizer on dataset ({desc})",
        fn_kwargs={"tokenizer": tokenizer, 
                "max_length": args.model_max_length, 
                "name_text": name_text},
        load_from_cache_file=False,
    )
    
    
    data_collator = DataCollatorForChat(
        tokenizer=tokenizer,
        mlm=False,
        start_token=args.start_token
    )
    
    return dataset, data_collator


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.stack([torch.argmax(logit, dim=-1) for logit in logits])
    return pred_ids, labels
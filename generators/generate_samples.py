import argparse
from lib import Datasets
from datasets import Dataset, DatasetDict


def create_dataset(dataset_names, num_examples, data_split="train", seed=42):
    examples = []

    for name in dataset_names:
        example = Datasets.LoadDataset(type=data_split, count=num_examples, name=name, seed=seed)
        examples.extend(example["dataset"])

    keys = examples[0].keys()
    data_dict = {key: [item[key] for item in examples] for key in keys}
    ds = Dataset.from_dict(data_dict)
    return ds


import argparse
import json
import ast
from datasets import Dataset, DatasetDict

def load_strings(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(raw)
            except Exception:
                data = raw.splitlines()

    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("Входной файл должен содержать список строк")
    return data

def main():
    parser = argparse.ArgumentParser(description='Load examples from multiple datasets.')
    parser.add_argument('--num_examples', type=int, default=2, help='Number of examples to load from each dataset')
    parser.add_argument('--data_split', type=str, default='train', help='Data split to use (e.g., train, validation, test)')
    parser.add_argument('--output_dir', type=str, default='./datasets/samples_dataset', help='Directory to save the combined dataset')
    parser.add_argument('--type', type=str, default='common-reasoning', help='Type of tasks')
    args = parser.parse_args()

    if args.type == 'common-reasoning':
        dataset_names = ["BoolQ", "PIQA", "SIQA", "hellaswag", "winogrande", "ARC-E", "ARC-C", "OBQA"]
    elif args.type == 'GSM8K':
        dataset_names = ['GSM8K']
    elif args.type == 'wiki2':
        texts = load_strings('./ready_datasets/wiki_256_4096.json')
        texts = texts[:args.num_examples]
        train_ds = Dataset.from_dict({"text": texts})
        ds_dict = DatasetDict({"train": train_ds})
        ds_dict.save_to_disk(args.output_dir, num_shards={'train': 2})
        print(ds_dict)
        return

    combined_dataset = create_dataset(dataset_names, args.num_examples, data_split=args.data_split, seed=69)

    combined_dataset_dict = DatasetDict({"train": combined_dataset})
    combined_dataset_dict.save_to_disk(args.output_dir, num_shards={'train': 2})

if __name__ == "__main__":
    main()
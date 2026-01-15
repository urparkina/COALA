import argparse
from datasets import load_dataset
from omegaconf import OmegaConf


parser = argparse.ArgumentParser(
        description='Loading from the HF hub and processing common reasoning dataset'
    )

parser.add_argument('config', help='Path to config file')
parser.add_argument('output', help='Path to output dir')


args = parser.parse_args()
config_pth = args.config
out_dir = args.output

dataset = load_dataset("wikitext", "wikitext-2-v1")

config = OmegaConf.load(config_pth)

dataset.save_to_disk(
    dataset_dict_path=out_dir,
    max_shard_size=config.get('max_shard_size', None),
    num_proc=config.get('num_proc', None),
)

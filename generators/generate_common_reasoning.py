import datasets
from datasets import load_dataset, DatasetDict

from transformers import AutoTokenizer
from omegaconf import OmegaConf
import functools
import argparse


def _get_BoolQ_instructions(example, tokenizer):
    instructions = [
        {"role": "system", "content": f"Please answer the following question with True or False. Follow the answer format, full answer not needed."},
        {"role": "user", "content": f"Question: {example['question']}\nAnswer format: True/False"},
    ]
    correct_answer = example['answer']

    instructions_ans = [
        {"role": "assistant", "content": f"The correct answer is {correct_answer}"}
    ]
    instructions_wa = [
        {"role": "assistant", "content": f"The correct answer is "}
    ]

    text = tokenizer.apply_chat_template(
        instructions + instructions_ans,
        tokenize=False
    )

    text_wa_answer = tokenizer.apply_chat_template(
        instructions + instructions_wa,
        tokenize=False
    )
    text_wa_answer = text_wa_answer.rsplit('<|eot_id|>', 1)[0]
    
    return {'text': text, 'text_wa_answer': text_wa_answer, 'correct_answer': str(correct_answer)}


def _get_PIQA_instructions(example, tokenizer):
    instructions = [
        {"role": "system", "content": f"Please choose the correct solution to the question. Follow the answer format, full answer not needed."},
        {"role": "user", "content": f"Question: {example['goal']}\nSolution1: {example['sol1']}\nSolution2: {example['sol2']}\nAnswer format: Solution1/Solution2"},
    ]
    correct_answer = f"Solution{example['label']+1}"

    instructions_ans = [
        {"role": "assistant", "content": f"The correct answer is {correct_answer}"}
    ]
    instructions_wa = [
        {"role": "assistant", "content": f"The correct answer is "}
    ]

    text = tokenizer.apply_chat_template(
        instructions + instructions_ans,
        tokenize=False
    )

    text_wa_answer = tokenizer.apply_chat_template(
        instructions + instructions_wa,
        tokenize=False
    )
    text_wa_answer = text_wa_answer.rsplit('<|eot_id|>', 1)[0]
    
    return {'text': text, 'text_wa_answer': text_wa_answer, 'correct_answer': str(correct_answer)}


def _get_SIQA_instructions(example, tokenizer):
    instructions = [
        {"role": "system", "content": f"Please choose the correct answer to the question based on the context provided. Follow the answer format, full answer not needed."},
        {"role": "user", "content": f"Context: {example['context']}\nQuestion: {example['question']}\nA: {example['answerA']}\nB: {example['answerB']}\nC: {example['answerC']}\nAnswer format: A/B/C"},
    ]
    correct_answer = f"{chr(int(example['label']) + ord('A') - 1)}"

    instructions_ans = [
        {"role": "assistant", "content": f"The correct answer is {correct_answer}"}
    ]
    instructions_wa = [
        {"role": "assistant", "content": f"The correct answer is "}
    ]

    text = tokenizer.apply_chat_template(
        instructions + instructions_ans,
        tokenize=False
    )

    text_wa_answer = tokenizer.apply_chat_template(
        instructions + instructions_wa,
        tokenize=False
    )
    text_wa_answer = text_wa_answer.rsplit('<|eot_id|>', 1)[0]
    
    return {'text': text, 'text_wa_answer': text_wa_answer, 'correct_answer': str(correct_answer)}


def _get_hellaswag_instructions(example, tokenizer):
    endings = '\n'.join(
        f'Ending{i}: {ending}'
        for i, ending in enumerate(example['endings'])
    )
    
    instructions = [
        {"role": "system", "content": f"Please choose the correct ending to complete the given sentence. Follow the answer format, full answer not needed."},
        {"role": "user", "content": f"{example['activity_label']}. {example['ctx']}\n{endings}\nAnswer format: Ending0/Ending1/Ending2/Ending3"},
    ]
    correct_answer = f"Ending{example['label']}"

    instructions_ans = [
        {"role": "assistant", "content": f"The correct answer is {correct_answer}"}
    ]
    instructions_wa = [
        {"role": "assistant", "content": f"The correct answer is "}
    ]

    text = tokenizer.apply_chat_template(
        instructions + instructions_ans,
        tokenize=False
    )

    text_wa_answer = tokenizer.apply_chat_template(
        instructions + instructions_wa,
        tokenize=False
    )
    text_wa_answer = text_wa_answer.rsplit('<|eot_id|>', 1)[0]
    
    return {'text': text, 'text_wa_answer': text_wa_answer, 'correct_answer': str(correct_answer)}


def _get_winogrande_instructions(example, tokenizer):
    instructions = [
        {"role": "system", "content": f"Please choose the correct answer to fill in the blank to complete the given sentence. Follow the answer format, full answer not needed."},
        {"role": "user", "content": f"Sentence: {example['sentence']}\nOption1: {example['option1']}\nOption2: {example['option2']}\nAnswer format: Option1/Option2"},
    ]
    correct_answer = f"Option{example['answer']}"

    instructions_ans = [
        {"role": "assistant", "content": f"The correct answer is {correct_answer}"}
    ]
    instructions_wa = [
        {"role": "assistant", "content": f"The correct answer is "}
    ]

    text = tokenizer.apply_chat_template(
        instructions + instructions_ans,
        tokenize=False
    )

    text_wa_answer = tokenizer.apply_chat_template(
        instructions + instructions_wa,
        tokenize=False
    )
    text_wa_answer = text_wa_answer.rsplit('<|eot_id|>', 1)[0]
    
    return {'text': text, 'text_wa_answer': text_wa_answer, 'correct_answer': str(correct_answer)}


def _get_ARC_instructions(example, tokenizer):
    answers = '\n'.join(
        f"{answer_label}: {answer_text}"
        for answer_label, answer_text in zip(example['choices']['label'], example['choices']['text'])
    )
    ans_format = '/'.join(
        f"{answer_label}"
        for answer_label in example['choices']['label']
    )
    
    instructions = [
        {"role": "system", "content": f"Please choose the correct answer to the question. Follow the answer format, full answer not needed."},
        {"role": "user", "content": f"Question: {example['question']}\n{answers}\nAnswer format: {ans_format}"},
    ]
    correct_answer = f"{example['answerKey']}"

    instructions_ans = [
        {"role": "assistant", "content": f"The correct answer is {correct_answer}"}
    ]
    instructions_wa = [
        {"role": "assistant", "content": f"The correct answer is "}
    ]

    text = tokenizer.apply_chat_template(
        instructions + instructions_ans,
        tokenize=False
    )

    text_wa_answer = tokenizer.apply_chat_template(
        instructions + instructions_wa,
        tokenize=False
    )
    text_wa_answer = text_wa_answer.rsplit('<|eot_id|>', 1)[0]
    
    return {'text': text, 'text_wa_answer': text_wa_answer, 'correct_answer': str(correct_answer)}


def _get_OBQA_instructions(example, tokenizer):
    answers = '\n'.join(
        f"{answer_label}: {answer_text}"
        for answer_label, answer_text in zip(example['choices']['label'], example['choices']['text'])
    )
    ans_format = '/'.join(
        f"{answer_label}"
        for answer_label in example['choices']['label']
    )
    
    instructions = [
        {"role": "system", "content": f"Please choose the correct answer to the question. Follow the answer format, full answer not needed."},
        {"role": "user", "content": f"Question: {example['question_stem']}\n{answers}\nAnswer format: {ans_format}"},
    ]
    correct_answer = f"{example['answerKey']}"

    instructions_ans = [
        {"role": "assistant", "content": f"The correct answer is {correct_answer}"}
    ]
    instructions_wa = [
        {"role": "assistant", "content": f"The correct answer is "}
    ]

    text = tokenizer.apply_chat_template(
        instructions + instructions_ans,
        tokenize=False
    )

    text_wa_answer = tokenizer.apply_chat_template(
        instructions + instructions_wa,
        tokenize=False,
        add_generation_prompt=False,
    )
    text_wa_answer = text_wa_answer.rsplit('<|eot_id|>', 1)[0]
    
    return {'text': text, 'text_wa_answer': text_wa_answer, 'correct_answer': str(correct_answer)}


def _load_datasets(config) -> list[DatasetDict]:
    BoolQ_dataset = load_dataset('google/boolq')
    PIQA_dataset = load_dataset('ybisk/piqa', trust_remote_code=True)
    SIQA_dataset = load_dataset('allenai/social_i_qa', trust_remote_code=True)
    hellaswag_dataset = load_dataset("Rowan/hellaswag")
    winogrande_dataset = load_dataset('allenai/winogrande', 'winogrande_debiased', trust_remote_code=True)
    ARC_e_dataset = load_dataset("allenai/ai2_arc", 'ARC-Easy')
    ARC_c_dataset = load_dataset("allenai/ai2_arc", 'ARC-Challenge')
    OBQA_dataset = load_dataset("allenai/openbookqa", "main")

    return [
        BoolQ_dataset, 
        PIQA_dataset, 
        SIQA_dataset, 
        hellaswag_dataset, 
        winogrande_dataset, 
        ARC_e_dataset, 
        ARC_c_dataset, 
        OBQA_dataset
    ]


def _add_answer_column(dataset: DatasetDict, dataset_name: str) -> DatasetDict:
    column_names = {
        'BoolQ': 'answer',
        'PIQA': 'label',
        'SIQA': 'label',
        'hellaswag': 'label',
        'winogrande': 'answer',
        'ARC-E': 'answerKey',
        'ARC-C': 'answerKey',
        'OBQA': 'answerKey',
    }

    ans_col = column_names[dataset_name]
    assert ans_col in dataset['validation'].features

    if ans_col not in dataset['test'].features:
        dataset['test'] = dataset['test'].add_column(
            name=ans_col,
            column='NO_ANSWER' * len(dataset['test'])
        )

    return dataset


def _process_datasets(config, dataset_ls: list[DatasetDict], tokenizer) -> list[DatasetDict]:
    dataset_processors = (
        _get_BoolQ_instructions,
        _get_PIQA_instructions,
        _get_SIQA_instructions,
        _get_hellaswag_instructions,
        _get_winogrande_instructions,
        _get_ARC_instructions,
        _get_ARC_instructions,
        _get_OBQA_instructions
    )

    dataset_names = (
        'BoolQ', 'PIQA', 'SIQA', 'hellaswag', 'winogrande', 'ARC-E', 'ARC-C', 'OBQA'
    )

    new_dataset_ls = []
    for dataset, dataset_name, processor in zip(dataset_ls, dataset_names, dataset_processors):   
        processor = functools.partial(
            processor,
            tokenizer=tokenizer,
        )

        dataset = dataset.map(
            processor, 
            batched=False, 
            num_proc=config.num_proc,
        )

        for split_name in dataset:
            dataset[split_name] = dataset[split_name].add_column(
                name='task',
                column=[dataset_name] * len(dataset[split_name])
            )
        
        new_dataset_ls.append(dataset)

    return new_dataset_ls


def _generate_dataset(config, tokenizer) -> DatasetDict:
    dataset_ls = _load_datasets(config=config)
    dataset_ls = _process_datasets(config=config, dataset_ls=dataset_ls, tokenizer=tokenizer)

    train_dataset = datasets.concatenate_datasets([
        dataset['train'].select_columns(['task', 'text', 'text_wa_answer', 'correct_answer'])
        for dataset in dataset_ls
    ])

    validation_dataset = datasets.concatenate_datasets([
        dataset['validation'].select_columns(['task', 'text', 'text_wa_answer', 'correct_answer'])
        for dataset in dataset_ls
    ])

    return DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset
    })


def _load_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, 
        **OmegaConf.to_object(config.tokenizer_config)
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def _load_and_process_task(config, tokenizer, task_name: str):
    dataset_loader = {
        'BoolQ':      lambda: load_dataset('google/boolq'),
        'PIQA':       lambda: load_dataset('ybisk/piqa', trust_remote_code=True),
        'SIQA':       lambda: load_dataset('allenai/social_i_qa', trust_remote_code=True),
        'hellaswag':  lambda: load_dataset("Rowan/hellaswag"),
        'winogrande': lambda: load_dataset('allenai/winogrande', 'winogrande_debiased', trust_remote_code=True),
        'ARC-E':      lambda: load_dataset("allenai/ai2_arc", 'ARC-Easy'),
        'ARC-C':      lambda: load_dataset("allenai/ai2_arc", 'ARC-Challenge'),
        'OBQA':       lambda: load_dataset("allenai/openbookqa", "main"),
    }
    dataset_processors = {
        'BoolQ':      _get_BoolQ_instructions,
        'PIQA':       _get_PIQA_instructions,
        'SIQA':       _get_SIQA_instructions,
        'hellaswag':  _get_hellaswag_instructions,
        'winogrande': _get_winogrande_instructions,
        'ARC-E':      _get_ARC_instructions,
        'ARC-C':      _get_ARC_instructions,
        'OBQA':       _get_OBQA_instructions
    }

    dataset = dataset_loader[task_name]()
    processor = dataset_processors[task_name]

    processor = functools.partial(
        processor,
        tokenizer=tokenizer,
    )

    dataset = dataset.map(
        processor, 
        batched=False, 
        num_proc=config.num_proc,
    )

    train_dataset =      dataset['train'].select_columns(['text', 'text_wa_answer', 'correct_answer'])
    validation_dataset = dataset['validation'].select_columns(['text', 'text_wa_answer', 'correct_answer'])
    
    return DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset
    })


def generate_task(config_pth, out_dir, task_name: str):
    config = OmegaConf.load(config_pth)

    tokenizer = _load_tokenizer(config)
    dataset: DatasetDict = _load_and_process_task(config, tokenizer, task_name)

    dataset.save_to_disk(
        dataset_dict_path=out_dir,
        max_shard_size=config.get('max_shard_size', None),
        num_proc=config.get('num_proc', None),
    )


def generate(config_pth, out_dir):
    config = OmegaConf.load(config_pth)

    tokenizer = _load_tokenizer(config)
    dataset: DatasetDict = _generate_dataset(config, tokenizer=tokenizer)

    dataset.save_to_disk(
        dataset_dict_path=out_dir,
        max_shard_size=config.get('max_shard_size', None),
        num_proc=config.get('num_proc', None),
    )


def main():
    parser = argparse.ArgumentParser(
        description='Loading from the HF hub and processing common reasoning dataset'
    )

    parser.add_argument('config', help='Path to config file')
    parser.add_argument('output', help='Path to output dir')
    parser.add_argument(
        '-s', '--select', 
        choices=['BoolQ', 'PIQA', 'SIQA', 'hellaswag', 'winogrande', 'ARC-E', 'ARC-C', 'OBQA'],
        help='Load only one task. Tasks: BoolQ, PIQA, SIQA, hellaswag, winogrande, ARC-E, ARC-C, OBQA'
    )

    args = parser.parse_args()
    cfg_path = args.config
    out_dir = args.output
    task_name = args.select

    if task_name is None:
        print(f"Loading common reasoning")
        generate(config_pth=cfg_path, out_dir=out_dir)
    else:
        print(f"Loading {task_name}")
        generate_task(config_pth=cfg_path, out_dir=out_dir, task_name=task_name)


if __name__ == '__main__':
    main()

import datasets
from tqdm import tqdm
import random
from rouge import Rouge
import pickle

import torch
import torch.utils
import torch.utils.data
import config
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 10
random.seed(SEED)
PROMPT_SYSTEM = 'You are a helpful assistant. '


rouge = Rouge()


def get_label(response, answers):
    if isinstance(answers, list):
        for answer in answers:
            scores = rouge.get_scores(response, answer)
            rougeL = scores[0]['rouge-l']['f']
            if rougeL > 0.3:
                return 0
            
    elif isinstance(answers, str):
        scores = rouge.get_scores(response, answers)
        rougeL = scores[0]['rouge-l']['f']
        if rougeL > 0.3:
            return 0
        
    return 1


def get_trivia_qa_answers(sample):
    answers = []
    answers.append(sample['answer']['value'])
    answers.append(sample['answer']['normalized_value'])
    answers.extend(sample['answer']['normalized_aliases'])
    answers.extend(sample['answer']['aliases'])

    return list(set(answers))


def get_trivia_qa_type(type, num_data):
    all_data = []

    if type not in ["validation", "train"]:
        print(f'error! invalid type: {type}.')
        return None

    data = datasets.load_dataset('trivia_qa', "rc.nocontext", split=type) 

    all_index = list(range(data.num_rows))
    sample_index = random.sample(all_index, num_data)
    shot_index = random.sample(list(set(all_index) - set(sample_index)), 10)

    data_samples = data.select(sample_index)
    shot_samples = data.select(shot_index)

    path = f'{config.output_dir}/trivia_qa/'
    if not os.path.exists(path):
        os.makedirs(path)

    # for sample in shot_samples:

    for sample in data_samples:
        d = {}
        d['question'] = sample['question']
        d['answers'] = get_trivia_qa_answers(sample)
        all_data.append(d)

    with open(f'{path}/{type}_og.pkl', 'wb') as w:
        pickle.dump(all_data, w)


def get_trivia_qa(num_test, num_train):
    get_trivia_qa_type('train', num_train)
    get_trivia_qa_type('validation', num_test)


def get_squad_type(type, num_data):
    if type not in ["validation", "train"]:
        print(f'error! invalid type: {type}.')
        return None

    data = datasets.load_dataset('rajpurkar/squad', split=type) 
    data_samples = data.select(random.sample(list(range(data.num_rows)), num_data)) 

    path = f'{config.output_dir}/squad/{type}'
    if not os.path.exists(path):
        os.makedirs(path)

    data_samples.save_to_disk(path)


def get_squad(num_test, num_train):
    get_squad_type('train', num_train)
    get_squad_type('validation', num_test)


def construct_prompt(data_name, question, shot_num):
    if data_name in ['trivia_qa']:
        prompt = f'{PROMPT_SYSTEM}Please answer the following question: {question}. Answer: '
        return prompt


def train_dataset_construction(model_name, device, data_name):
    results = []
    path = f'/mnt/local/wxy/models/{model_name}'
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype = torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path)

    path = f'{config.output_dir}/{data_name}/train'
    train_dataset = datasets.load_from_disk(path).select(range(3))

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    with torch.no_grad():
        max_length = 128
        for sample in tqdm(dataloader):
            d = {}
            question = sample['question']
            prompt = construct_prompt(data_name, question, 0)
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            output = model.generate(**inputs
                                    , do_sample = False
                                    , max_new_tokens=max_length
                                    )
            
            response_content = tokenizer.decode(output[0][len(inputs.input_ids):], skip_special_tokens=True)
            answers = sample['answers']
            label = get_label(response_content, answers)
            d['prompt'] = prompt
            d['question'] = question
            d['response'] = response_content
            d['label'] = label
            d['answers'] = answers
            results.append(d)

    output_path = f'{config.output_dir}/{data_name}/labeled_data'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(f'{output_path}/train.pkl', 'wb') as w:
        pickle.dump(results, w)
        
        
def test_dataset_construction(model_name, device, data_name):
    """
    构造测试集：
    - 从 validation 集取样
    - 用 greedy decoding 生成回答
    - 判断模型回答是否与标准答案完全一致（label）
    """
    results = []
    path = f'/mnt/local/wxy/models/{model_name}'
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path)

    # 加载 validation 数据集
    path = f'{config.output_dir}/{data_name}/validation'
    test_dataset = datasets.load_from_disk(path).select(range(3))  # 示例取3条，可自行改

    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    with torch.no_grad():
        max_length = 128
        for sample in tqdm(dataloader):
            d = {}
            question = sample['question']
            prompt = construct_prompt(data_name, question, 0)
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            output = model.generate(**inputs, do_sample=False, max_new_tokens=max_length)

            response_content = tokenizer.decode(output[0][len(inputs.input_ids):], skip_special_tokens=True)
            answers = sample['answers']

            # 🔹 label: greedy response 是否与任一 ground truth 完全一致
            label = 1  # 默认错误
            if isinstance(answers, list):
                for ans in answers:
                    if response_content.strip().lower() == ans.strip().lower():
                        label = 0
                        break
            elif isinstance(answers, str):
                if response_content.strip().lower() == answers.strip().lower():
                    label = 0

            d['prompt'] = prompt
            d['question'] = question
            d['response'] = response_content
            d['label'] = label
            d['answers'] = answers
            results.append(d)

    # 保存结果
    output_path = f'{config.output_dir}/{data_name}/labeled_data'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(f'{output_path}/test.pkl', 'wb') as w:
        pickle.dump(results, w)

if __name__ == '__main__':
    get_trivia_qa(300, 1500)
    # get_squad(300, 1500)
    train_dataset_construction('vicuna-7b', 'cuda:1', 'trivia_qa')
    test_dataset_construction('vicuna-7b', 'cuda:1', 'trivia_qa')
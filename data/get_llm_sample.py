from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoTokenizer, AutoModel, DataCollatorWithPadding, AutoModelForCausalLM, LlamaForCausalLM
import os
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import transformers
import torch.nn as nn

def add_role(batch_list):
    question = [iterm['question'] for iterm in batch_list]
    answers = [iterm['answer'] for iterm in batch_list]
    q_id = [iterm['question_id'] for iterm in batch_list]

    return q_id, question, answers

def add_role(batch_list):
    question = [iterm['question'] for iterm in batch_list]
    label = [iterm['label'] for iterm in batch_list]
    q_id = [iterm['q_id'] for iterm in batch_list]

    return q_id, question, label

def add_role(batch_list):
    categorys = [iterm['category'] for iterm in batch_list]
    types = [iterm['type'] for iterm in batch_list]
    questions = [iterm['question'] for iterm in batch_list]
    best_answers = [iterm['best_answer'] for iterm in batch_list]
    correct_answers = [iterm['correct_answers'] for iterm in batch_list]
    incorrect_answers = [iterm['incorrect_answers'] for iterm in batch_list]
    sources = [iterm['source'] for iterm in batch_list]

    return categorys, types, questions, best_answers, correct_answers, incorrect_answers, sources

def add_role(batch_list):
    question = [iterm['question'] for iterm in batch_list]
    answers = [iterm['label'] for iterm in batch_list]

    return question, answers


def add_role_hotpot_qa_has_support(batch_list):
    ids = [iterm['id'] for iterm in batch_list]
    answers = [iterm['answer'] for iterm in batch_list]
    questions = [iterm['question'] for iterm in batch_list]
    types = [iterm['type'] for iterm in batch_list]
    level = [iterm['level'] for iterm in batch_list]
    supporting_facts = [iterm['supporting_facts'] for iterm in batch_list]

    return ids, questions,answers, types , level, supporting_facts

def add_role_hotpot_qa(batch_list):
    ids = [iterm['id'] for iterm in batch_list]
    answers = [iterm['answer'] for iterm in batch_list]
    questions = [iterm['question'] for iterm in batch_list]
    types = [iterm['type'] for iterm in batch_list]
    level = [iterm['level'] for iterm in batch_list]

    return ids, questions,answers, types , level


def add_role_gsm8k(batch_list):
    question = [iterm['question'] for iterm in batch_list]
    answers = [iterm['answer'] for iterm in batch_list]

    return question, answers


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# gpu = 0
batch_size = 8

# model_name = 'Mistral-7B-Instruct-v0.2'
model_name = 'vicuna-33b'

# model_path = f'/mnt/local/xywang/models/{model_name}'
model_path = f'/home/wxy/models/{model_name}'
# device = torch.device("cuda:0,1")

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="sequential")
# model = AutoModelForCausalLM.from_pretrained('lmsys/vicuna-33b-v1.3', device_map="sequential")
# model.to('cuda')
# model.cuda('cuda:1,0')
model = model.to(torch.float16)
model = model.eval()

# device = torch.device("cuda:0,1" if torch.cuda.is_available() else "cpu") 
# model= nn.DataParallel(model,device_ids = [0, 1])
# model.to(device)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# device = torch.device("cuda:0")
# model.to(device)
# if torch.cuda.device_count() > 1:
# 	model= nn.DataParallel(model,device_ids=[0,1])


tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token


# data_name = 'hotpot_qa'
# file_type = 'test_e_m_h_has_support'


data_name = 'gsm8k'
# file_type = 'train_816_1500'
data_name = 'hotpot_qa'
# file_type = 'test_e_m_h_has_support'
file_type = 'test_e_m_h_has_support_336'

if data_name == 'gsm8k':
    shot = '0shot'
    data_path_sub = f'{data_name}/{shot}' if shot else f'{data_name}'
    output_path_seed = shot
    output_path = f'./datasets/{data_path_sub}/output/{output_path_seed}'
elif data_name == 'hotpot_qa':
    data_path_sub = f'{data_name}'
    output_path = f'./datasets/{data_path_sub}/output'

data_path = f'/home/wxy/project/hallucination_predict_by_prompt/data/datasets/{data_path_sub}/{file_type}.json' 
dataset = load_dataset(path='json', data_files=data_path)


if not os.path.exists(output_path):
    os.makedirs(output_path)

# new_file_type = file_type.replace('_has_support', '')
new_file_type = file_type

f = open(f'{output_path}/{new_file_type}_{model_name}_1_sample.json', 'a')

if 'gsm8k' in data_name:
    dataloader = DataLoader(dataset=dataset['train'], batch_size=batch_size, shuffle=False, collate_fn=add_role_gsm8k)

elif 'hotpot' in data_name:
    if 'no_support' in data_name:
        dataloader = DataLoader(dataset=dataset['train'], batch_size=batch_size, shuffle=False, collate_fn=add_role_hotpot_qa)
    else:
        dataloader = DataLoader(dataset=dataset['train'], batch_size=batch_size, shuffle=False, collate_fn=add_role_hotpot_qa_has_support)
    # if 'has_support' not in file_type:
    #     dataloader = DataLoader(dataset=dataset['train'], batch_size=batch_size, shuffle=False, collate_fn=add_role_hotpot_qa)
    # else:
        # dataloader = DataLoader(dataset=dataset['train'], batch_size=batch_size, shuffle=False, collate_fn=add_role_hotpot_qa_has_support)

# file_type = 'test'
# data_path = f'/root/autodl-fs/dataset/trivia_qa/v1/{file_type}_v1.json'
# data_path = '/root/autodl-fs/dataset/trivia_qa/v1/trivia_qa_v1_test_standard.json'
# data_path = '/root/autodl-fs/dataset/trivia_qa/train_standard_7939.json'
# data_path = '/root/autodl-fs/dataset/gsm8k/v2/test.json'
# data_path = '/root/autodl-fs/dataset/gsm8k/v2/train.json'

# data_path = '/root/autodl-fs/dataset/truthful_qa/test.json'
# data_path = '/root/autodl-fs/dataset/primer_data/train_v2.json'

# data_path = f'/root/autodl-fs/dataset/primer_data/{file_type}.json'


# f = open(f'/root/autodl-fs/dataset/trivia_qa/llama_result/train_standard_7939_1_sample.json', 'w')
# f = open(f'/root/autodl-fs/dataset/truthful_qa/test_llama_7b_1_smaples.json', 'w')

# f = open(f'/root/autodl-fs/dataset/primer_data/{file_type}_llama_7b_5_smaples.json', 'w')
# f = open(f'/root/autodl-fs/dataset/primer_data/train_v2_llama_7b_1_smaples.json', 'w')

def get_prompt(data_name, batch):
    list_prompt = []
    if 'gsm8k' in data_name:
        questions,answers = batch
        for i in range(len(questions)):
            question = questions[i]
            prompt = f'You are a helpful assistant. Please answer the question. \n\n QUESTION:{question} \n\n ANSWER: '
            list_prompt.append(prompt)

        return questions, answers, list_prompt 

    if 'hotpot_qa' in data_name:
        if len(batch) == 6:
            ids, questions,answers, types , level, supporting_facts = batch
            for i in range(len(questions)):
                question = questions[i]
                supporting_fact = supporting_facts[i]
                prompt = f'You are a helpful assistant. Please answer the question according to the given supporting facts. \n\n SUPPORT_FACTS:{supporting_fact} \n\n QUESTION:{question} \n\n Answer: '
                list_prompt.append(prompt)
            return ids, questions,answers, types , level, supporting_facts, list_prompt
        elif len(batch) == 5:
            ids, questions,answers, types , level = batch
            for i in range(len(questions)):
                question = questions[i]
                prompt = f'You are a helpful assistant. Please answer the question. \n\n QUESTION:{question} \n\n ANSWER: '
                list_prompt.append(prompt)

            return ids, questions,answers, types , level, list_prompt 

for batch in tqdm(dataloader):
    # q_ids, questions, labels = batch


    if 'gsm8k' in data_name:
        questions, answers, list_prompt = get_prompt(data_name, batch)

    elif 'hotpot_qa' in data_name:
        if 'no_support' in data_name:
            ids, questions,answers, types , level, list_prompt = get_prompt(data_name, batch)
        else:
            ids, questions,answers, types , level, supporting_facts, list_prompt = get_prompt(data_name, batch)

    # categorys, types, questions, best_answers, correct_answers, incorrect_answers, sources = batch

    input_tokenized = tokenizer(list_prompt, return_tensors='pt', padding=True).to('cuda')
    generate_input = {
        "input_ids": input_tokenized.input_ids
        , "attention_mask": input_tokenized.attention_mask
        , "eos_token_id": tokenizer.eos_token_id
        , "bos_token_id": tokenizer.bos_token_id
        , "pad_token_id": tokenizer.pad_token_id
    }

    generate_ids = model.generate(
            **generate_input
            , do_sample=False
            , max_new_tokens = 512
        )

    for i in range(len(questions)):
        d = {}
        tmp_question = list_prompt[i]

        if 'gsm8k' in data_name:
            d['question'] = questions[i]
            d['answer'] = answers[i]
            d['prompt'] = tmp_question

        if 'hotpot_qa' in data_name:
            d['id'] = ids[i]
            d['question'] = questions[i]
            d['prompt'] = list_prompt[i]
            d['answer'] = answers[i]
            d['type'] = types[i]
            d['level'] = level[i]
            if 'no_support' not in data_name:
                d['supporting_fact'] = supporting_facts[i]

        # d['predict'] = sequences[i]['generated_text'].replace(tmp_question, '')
        d['predict'] = tokenizer.decode(generate_ids[i], skip_special_tokens=True).replace(tmp_question, '')


        f.write(json.dumps(d, ensure_ascii=False)+'\n')

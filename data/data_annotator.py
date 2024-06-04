# gpt4 打标

import os
import openai
import numpy as np
from tqdm import tqdm
import time
from openai._client import OpenAI
import json

API_KEY = 'sk-4yxV5aMhnwe6R1v395B301B777A5491fA02474622fF9F079'
client = OpenAI(api_key=API_KEY, base_url="https://www.jcapikey.com/v1")

# api_key = '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'
# client = OpenAI(api_key=api_key)

# PROMPT_TEMPLATE = "Question: {question} \n\nSuggested answer:{correct_answer} \n\nMy Answer: {predict_answer}\n\nPlease check wether my answer is correct. Answer Yes or No.\n\nAnswer: "
PROMPT_TEMPLATE = "Question: {question} \n\nAnswer: {predict_answer}\n\nPlease check wether the given Answer is correct to the question. Answer Yes or No.\n\nAnswer: "

def get_gpt_response(question, predict_answer, correct_answer=None):
    # prompt = PROMPT_TEMPLATE.format(question=question, correct_answer=correct_answer, predict_answer=predict_answer)
    prompt = PROMPT_TEMPLATE.format(question=question, predict_answer=predict_answer)

    response = client.chat.completions.create(
                    model='gpt-4-turbo-2024-04-09',
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0, # 0.0 = deterministic
                    max_tokens=10, # max_tokens is the generated one,
                )
    output_text = response.choices[0].message.content
    generate_text = output_text.replace(prompt, "")
    # time.sleep(1)
    return generate_text

def get_label_from_gpt_response(response):
    response = response.rstrip().strip().lower()
    if response[:2] == 'no':
        return 1
    if response[:3] == 'yes':
        return 0
    
    return None

def get_label_lexical_matching(predict, answer):
    answer = answer.strip().replace('.', '')
    predict = predict.rstrip().strip().replace('\n', '')
    predict_1 = predict.replace('.', '')
    predict_2 = predict.replace(',', '')
    label = None

    if answer.lower() in [predict.lower(), predict_1.lower(), predict_2.lower()]:
        label = 0
        # w.write(json.dumps(data, ensure_ascii=False)+'\n')
    elif answer.lower() in predict.lower():
        label = 1
    else:
        label = 2

    return label


''' 
处理 1 sample
'''

if __name__ == '__main__':
    # model_name = 'llama2-7b-chat-hf'
    # model_name = 'vicuna-33b'
    # model_name = 'vicuna-13b'
    model_name = 'Mistral-7B-Instruct-v0.2'

    sample = '1_sample'
    
    # data_type = 'train_all'
    data_type = 'train_0_1500'
    # hotpot_support = 'has_support'
    # hotpot_support = 'no_support'

    # hotpot_qa
    parent_path = './datasets/gsm8k/0shot/output/0shot/'
    input_file_path = f'{parent_path}/{data_type}_{model_name}_{sample}.json'
    output_file_path = f'{parent_path}/{data_type}_{model_name}_{sample}_has_label.json'

    # gsm8k
    # parent_path = '/Users/ganning/Documents/project_python/hallucination_detect_explore/dataset/gsm8k_/v3_2000'
    # input_file_path = f'{parent_path}/llm_output/{data_type}_{model_name}_{sample}.json'
    # output_file_path = f'{parent_path}/llm_output_label/{data_type}_{model_name}_{sample}_label.json'

    output_file_path_parent = '/'.join(output_file_path.split('/')[:-1])
    if not os.path.exists(output_file_path_parent):
        os.makedirs(output_file_path_parent)

    with open(input_file_path, 'r') as f, open(output_file_path, 'a') as w:
        for line in tqdm(f.readlines()[787:1000]):
            d = json.loads(line.strip())
            question = d['question']
            answer = d['answer'].lower()
            predict = d['predict']
            predict1 = d['predict'].rstrip().strip().lower()
            predict2 = d['predict'].rstrip().strip().replace('.', '').lower()
            predict3 = d['predict'].rstrip().strip().replace(',', '').lower()
            predict4 = d['predict'].rstrip().strip().replace('.', '').replace(',', '').lower()
            # d['id'] = d_id[question]
            if answer in [predict, predict1, predict2, predict3, predict4]:
                d['label'] = 0
                d['label_type'] = 'same'
                w.write(json.dumps(d, ensure_ascii=False)+'\n')
            else:
                generate_text = get_gpt_response(question , predict, answer)
                d['label'] = get_label_from_gpt_response(generate_text)
                d['label_type'] = 'gpt4'+'_'+generate_text
                w.write(json.dumps(d, ensure_ascii=False)+'\n') 
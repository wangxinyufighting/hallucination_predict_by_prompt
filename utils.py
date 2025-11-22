from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch
import json
from tqdm import tqdm
import os
from vllm import LLM, SamplingParams


from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import Dict, List

from einops import rearrange
from tqdm import tqdm
import numpy as np
import torch
import numpy as np
import torch
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import sklearn
from sklearn.metrics import accuracy_score,\
    classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, precision_recall_curve, recall_score, auc
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import json
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset
import time
from sklearn.neural_network import MLPClassifier



# --- 1. 定义核心功能函数 ---

def get_and_save_representations(
    prompt: str,
    model,
    tokenizer,
    # save_dir: str = "model_representations"
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    获取并保存一个 prompt 在模型中每一层和每个注意力头的隐层表示。

    Args:
        prompt (str): 输入的文本。
        model: 预加载的 Hugging Face 模型。
        tokenizer: 预加载的 Hugging Face 分词器。

    Returns:
        dict: 包含捕获到的隐层表示的字典。
    """
    # 确保模型处于评估模式
    model.eval()

    # --- 数据存储结构 ---
    # 存储每一层（DecoderLayer）最终输出的隐层状态
    layer_hidden_states: Dict[int, torch.Tensor] = {}
    # 存储每一层自注意力模块（self-attention）的输出
    attn_outputs: Dict[int, torch.Tensor] = {}

    # --- Hook 定义 ---
    # 使用闭包/工厂模式，让每个 hook 函数能“记住”自己的层索引
    def create_layer_hook(layer_idx: int):
        def layer_hook_fn(module, input, output):
            # Qwen2DecoderLayer的输出是一个元组，第一个元素是hidden_states
            layer_hidden_states[layer_idx] = output[0].detach().cpu()
        return layer_hook_fn

    def create_attn_hook(layer_idx: int):
        def attn_hook_fn(module, input, output):
            # Qwen2Attention的输出也是一个元组，第一个元素是注意力头的拼接输出
            attn_outputs[layer_idx] = output[0].detach().cpu()
        return attn_hook_fn

    # --- 注册 Hooks ---
    hook_handles: List = []
    num_layers = model.config.num_hidden_layers
    # print(f"模型总共有 {num_layers} 层，将为每一层注册 hooks...")

    for i in range(num_layers):
        # 1. 为整个 Transformer Block (DecoderLayer) 注册 hook
        layer_module = model.model.layers[i]
        handle = layer_module.register_forward_hook(create_layer_hook(i))
        hook_handles.append(handle)

        # 2. 为自注意力模块 (self_attn) 注册 hook
        attn_module = model.model.layers[i].self_attn
        handle = attn_module.register_forward_hook(create_attn_hook(i))
        hook_handles.append(handle)

    # --- 执行前向传播 ---
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # print(f"\n正在为 prompt: '{prompt}' 执行前向传播...")
    with torch.no_grad():
        model(**inputs)

    # --- 移除所有 Hooks (非常重要！) ---
    for handle in hook_handles:
        handle.remove()
    # print("\n所有 hooks 已被成功移除。")

    # --- 后处理注意力头输出 ---
    # print("正在处理捕获到的注意力头输出...")
    config = model.config
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    all_head_representations = []
    for layer_idx, attn_output in attn_outputs.items():
        batch_size, seq_len, _ = attn_output.shape
        # 将 (batch, seq_len, hidden_size) -> (batch, seq_len, num_heads, head_dim)
        reshaped_output = attn_output.view(batch_size, seq_len, num_heads, head_dim)
        all_head_representations.append(reshaped_output[:, -1, :, :])
    
    all_head_representations = np.array(all_head_representations)  # (Layers, Heads, Head_Dim)

    all_layer_hidden_states = []
    for layer_idx, layer_hidden_state in layer_hidden_states.items():
        print(layer_hidden_state.shape)
        all_layer_hidden_states.append(layer_hidden_state[:, -1, :])
    all_layer_hidden_states = np.array(all_layer_hidden_states)  # (Layers, Tokens, Hidden_Size)
    all_layer_hidden_states = np.array(all_layer_hidden_states)

    final_data = {
        "prompt": prompt,
        "layer_hidden_states": all_layer_hidden_states,
        "all_head_representations": all_head_representations
    }

    return final_data

def get_prompts(data_name, data_case):
    prompt = None
    
    if data_name.lower() == 'gsm8k':
        question = data_case['question']
        prompt = "Let's think step by step and output the final answer in \\boxed{}."
        prompt = f'{question}\n{prompt}'
    if 'mmlu' in data_name.lower():
        question = data_case['question']
        options = []
        for index, option in enumerate(data_case['options']):
            options.append(chr(ord('A') + index)+'. '+option)
        options = '\n'.join(options) 
        prompt = """
            Please Reason about the correct answer based on the question and options provided. 
            After your reasoning, you will select the most correct OPTION(A, B, C, D, F, G, H, I, J) and write the OPTION in \\boxed{}. 
            For example: \\boxed{A}
            """
        prompt = f'Question:\n{question}\nOptions:{options}\n{prompt}'

    return prompt

def get_answer_extractor(data_name):
    if 'gsm' in data_name.lower(): 
        return gsm8k_answer_extractor
    elif 'mmlu' in data_name.lower():
        return multi_choice_question_answer_extractor

def get_messages(prompt, tokenizer):
    # 构建对话格式
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return text

def multi_choice_question_answer_extractor(solution_str, prompt=""):
    if '\boxed' in solution_str:
        solution_str = solution_str.replace('\\boxed', 'boxed')
    if 'Boxed' in solution_str:
        solution_str = solution_str.replace('Boxed', 'boxed')

    solution = re.search("(?<=boxed{)[a-zA-Z]", solution_str)

    final_solution = ""
    if solution is not None:
        final_solution = solution.group(0)
    else:
        if 'answer is:\n\n' in solution_str:
            # print(solution_str)
            solution = re.search("(?<=answer is:\n\n)[a-zA-Z]", solution_str)
            if solution is not None:
                final_solution = solution.group(0)
        elif 'answer is: ' in solution_str:
            solution = re.search("(?<=answer is: )\w", solution_str)
            if solution is not None:
                final_solution = solution.group(0)
        elif 'answer is ' in solution_str:
            solution = re.search("(?<=answer is )\w", solution_str)
            if solution is not None:
                final_solution = solution.group(0)
        elif 'Answer: ' in solution_str:
            solution = re.search("(?<=Answer\: )\w", solution_str)
            if solution is not None:
                final_solution = solution.group(0)
        elif 'answer: ' in solution_str:
            solution = re.search("(?<=answer: )\w", solution_str)
            if solution is not None:
                final_solution = solution.group(0)
        elif 'Answer\n' in solution_str:
            solution = re.search("(?<=Final Answer\n)[a-zA-Z]", solution_str)
            if solution is not None:
                final_solution = solution.group(0)
        elif 'answer is:\n' in solution_str:
            solution = re.search("(?<=answer is:\n)[a-zA-Z]", solution_str)
            if solution is not None:
                final_solution = solution.group(0)
        elif 'Final Answer**: ' in solution_str:
            solution = re.search("(?<=Final Answer\*\*: )\w", solution_str)
            if solution is not None:
                final_solution = solution.group(0)
        elif solution_str in prompt:
            final_solution = solution_str
        elif 'which is option ' in solution_str:
            solution = re.search("(?<=which is option )[a-zA-Z]", solution_str)
            if solution is not None:
                final_solution = solution.group(0)
        elif 'boxed: ' in solution_str:
            solution = re.search("(?<=boxed: )[a-zA-Z]", solution_str)
            if solution is not None:
                final_solution = solution.group(0)
        elif 'is **' in solution_str:
            solution = re.search("(?<=is \*\*)[a-zA-Z]", solution_str)
            if solution is not None:
                final_solution = solution.group(0)
        else:
            solution = re.search("^[a-zA-Z]\.", solution_str)
            if solution is not None:
                final_solution = solution.group(0)[0]
            else:
                final_solution = ""

    return final_solution

import re
def gsm8k_answer_extractor(solution_str):
    solution = re.search("(?<=boxed{)-?\d+(?:[.,]\d+)*(?=\})", solution_str)

    if solution is not None:
        final_solution = solution.group(0)
    else:
        final_solution = "-1000000000"

    return final_solution

from collections import Counter


def get_majority_answer(sample_answers):
    answer_counter = Counter(sample_answers)
    majority_answer = Counter(sample_answers).most_common(1)[0][0] 
    return majority_answer

def is_right_gsm(greedy_answer:str, majority_answer):

    return float(greedy_answer) == float(majority_answer)

def is_right_multi_choices_question(pred:str, gt) -> float:

    if pred and gt:
        gt = gt.strip()
        pred = pred.strip()
        while pred[-1] == '.':
            pred = pred[:-1]

        return gt.lower() == pred.lower()
        
    return False

def get_label(greedy_answer:str, sample_answers:list[str], data_name):

    majority_answer = get_majority_answer(sample_answers)

    if 'gsm' in data_name.lower(): 
        return is_right_gsm(greedy_answer, majority_answer)
    elif 'mmlu' in data_name.lower():
        return is_right_multi_choices_question(greedy_answer, majority_answer)


def _generate(model, tokenizer, prompt, max_new_tokens=512, num_samples=1, temperature=0.7):
    text = get_messages(prompt, tokenizer)
    # 编码输入 - 复制成batch
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    do_sample = True if temperature > 0 else False

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample, 
            num_return_sequences=num_samples,  
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    responses = []
    for i in range(num_samples):
        # 解码单个输出，只保留生成的部分
        response = tokenizer.decode(
            outputs[i][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        responses.append(response)

    return responses


def get_labels(model_name, data_name, dataset_path, num_samples=10, temperature=0.7, max_new_tokens=512):
    model_path = f'/mnt/local/wxy/models/{model_name}'
    
    # 使用 vLLM 进行加速
    print(f"Initializing vLLM with model: {model_path}")
    # tensor_parallel_size 设置为 GPU 数量
    llm = LLM(model=model_path, trust_remote_code=True, tensor_parallel_size=torch.cuda.device_count())
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    data_responses_output_path = f'./datasets/{data_name}/responses'
    if not os.path.exists(data_responses_output_path):
        os.makedirs(data_responses_output_path)

    answer_extractor = get_answer_extractor(data_name)

    data_file_name = dataset_path.split('/')[-1].split('.')[0]
    data_responses_output_file = f'{data_responses_output_path}/responses_{data_file_name}.json'

    # 读取所有数据
    with open(dataset_path, 'r') as f:
        lines = f.readlines()
    
    all_data = [json.loads(line.strip()) for line in lines]
    prompts = []
    raw_prompts = []
    questions = []

    print("Preparing prompts...")
    for data in tqdm(all_data):
        question = data['question']
        raw_prompt = get_prompts(data_name, data)
        # 应用聊天模板
        text = get_messages(raw_prompt, tokenizer)
        
        prompts.append(text)
        raw_prompts.append(raw_prompt)
        questions.append(question)

    # 1. Greedy Generation
    print("Generating greedy responses...")
    sampling_params_greedy = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    outputs_greedy = llm.generate(prompts, sampling_params_greedy)

    # 2. Sampling Generation
    print(f"Generating sampled responses (n={num_samples})...")
    sampling_params_sample = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_new_tokens, n=num_samples)
    outputs_sample = llm.generate(prompts, sampling_params_sample)

    print(f"Saving results to {data_responses_output_file}...")
    with open(data_responses_output_file, 'w') as w:
        for i in range(len(all_data)):
            greedy_output = outputs_greedy[i]
            sample_output = outputs_sample[i]
            
            greedy_response = greedy_output.outputs[0].text
            sample_responses = [o.text for o in sample_output.outputs]
            
            greedy_answer = answer_extractor(greedy_response)
            sample_answers = [answer_extractor(ans) for ans in sample_responses]

            new_data = {}
            new_data['question'] = questions[i]
            new_data['prompt'] = raw_prompts[i]
            new_data['greedy_response'] = greedy_response
            new_data['greedy_answer'] = greedy_answer
            new_data['sample_responses'] = sample_responses
            new_data['sample_answers'] = sample_answers
            new_data['label'] = get_label(greedy_answer, sample_answers, data_name)

            w.write(json.dumps(new_data, ensure_ascii=False)+'\n')

    # 清理显存，防止后续加载 HF 模型 OOM
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return data_responses_output_file

def save_hidden_states(args):
    model_name=args.model_name
    data_name=args.data_name
    dataset_path=args.dataset_path
    temperature=args.temperature
    num_samples=args.num_samples
    max_new_tokens=args.max_new_tokens
     
    # 优先检查是否需要生成标签，避免重复加载模型导致OOM
    data_responses_output_file = f'/mnt/local2/wxy/hallucination_predict_by_prompt/datasets/{data_name}/responses.json' 
    if not os.path.exists(data_responses_output_file):
        print("Responses file not found, generating labels first...")
        data_responses_output_file = get_labels(
            model_name=model_name
            , data_name=data_name
            , dataset_path=dataset_path
            , temperature=temperature
            , num_samples=num_samples
            , max_new_tokens=max_new_tokens)
    model_path = f'/mnt/local/wxy/models/{model_name}'

    print("正在加载模型和分词器 (HF)...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("模型加载完成。")

    all_hidden = []
    all_attention = []
    all_label = []

    print(f'load data from {data_responses_output_file}')

    with open(data_responses_output_file, 'r') as f:
        for line in tqdm(f.readlines()):
            data = json.loads(line.strip())
            prompt = data['prompt']
            # --- 调用函数获取并保存表示 ---
            captured_data = get_and_save_representations(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer
            )

            # print(captured_data["layer_hidden_states"].shape)
            # print(captured_data["all_head_representations"].shape)

            all_hidden.append(captured_data["layer_hidden_states"])
            all_attention.append(captured_data["all_head_representations"])
            all_label.append(data['label'])

    path = './feature'
    if not os.path.exists(path):
        os.makedirs(path)

    # all_hidden shape: (Num_Examples, Layers, 1, Hidden_Size)]
    # all_attention shape: (Num_Examples, Layers, 1, Heads, Head_Dim)
    # all_label shape: (Num_Examples,)

    np.save(f'{path}/{data_name}_{model_name}_layers.npy', all_hidden)
    np.save(f'{path}/{data_name}_{model_name}_heads.npy', all_attention)
    np.save(f'{path}/{data_name}_{model_name}_labels.npy', all_label)

    print(f'layer hidden states saved in {path}/{data_name}_{model_name}_layers.npy')
    print(f'head hidden states saved in {path}/{data_name}_{model_name}_heads.npy')
    print(f'label saved in {path}/{data_name}_{model_name}_labels.npy')


def probe(layer_num, head_num, head_wise_activations_train, labels_train, head_wise_activations_test, labels_test, max_iter=5):
    m_auc = np.empty([layer_num,head_num], dtype = float) 
    m_auroc = np.empty([layer_num,head_num], dtype = float) 

    d_clf = {}
    for layer in tqdm(range(layer_num)):
        for head in range(head_num):
            X_train = head_wise_activations_train[:, layer, head, :]
            Y_train = labels_train
            assert X_train.shape[0]==Y_train.shape[0]

            clf = LogisticRegression(max_iter=max_iter).fit(X_train, Y_train) 

            X_test = head_wise_activations_test[:, layer, head, :]
            Y_test = labels_test
            assert X_test.shape[0]==Y_test.shape[0]

            if layer not in d_clf:
                d_clf[layer] = {}
            if head not in d_clf[layer]:
                d_clf[layer][head] = clf

            tempPredicts = clf.predict(X_test)
            tempLogits = [i[1] for i in clf.predict_proba(X_test)]

            precision_list, recall_list, _ = precision_recall_curve(Y_test, tempLogits)
            auroc = roc_auc_score(Y_test, tempLogits)
            prauc = auc(recall_list, precision_list)

            m_auc[layer][head] = prauc
            m_auroc[layer][head] = auroc

    return m_auc, m_auroc, d_clf

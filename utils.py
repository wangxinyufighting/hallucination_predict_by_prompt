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
        all_layer_hidden_states.append(layer_hidden_state[:, -1, :])
    all_layer_hidden_states = np.array(all_layer_hidden_states)  # (Layers, Tokens, Hidden_Size)
    all_layer_hidden_states = np.array(all_layer_hidden_states)

    final_data = {
        "prompt": prompt,
        "layer_hidden_states": all_layer_hidden_states,
        "all_head_representations": all_head_representations
    }

    return final_data

def get_prompts(data_name):
    prompt = None
    
    if data_name.lower() == 'gsm8k':
        prompt = "Let's think step by step and output the final answer in \\boxed{}."

    return prompt

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

import re
def gsm8k_answer_extractor(solution_str):
    solution = re.search("(?<=boxed{)-?\d+(?:[.,]\d+)*(?=\})", solution_str)

    if solution is not None:
        final_solution = solution.group(0)
    else:
        final_solution = "-1000000000"

    return final_solution

from collections import Counter
def get_label(greedy_answer:str, sample_answers:list[str]):
    answer_counter = Counter(sample_answers)

    if len(answer_counter) == len(sample_answers):
        return False

    majority_answer = Counter(sample_answers).most_common(1)[0][0] 

    return float(greedy_answer) == float(majority_answer)


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

def get_labels_vllm(model_name, data_name, dataset_path, num_samples=10, temperature=0.7, max_new_tokens=512):
    """
    使用 VLLM 框架为数据集生成标签。

    Args:
        model_name (str): 要加载的模型名称（位于'/mnt/local/wxy/models/'下）。
        data_name (str): 数据集名称，用于获取 prompt 和设置输出路径。
        dataset_path (str): 输入数据集的 .jsonl 文件路径。
        num_samples (int): 每个 prompt 需要采样的响应数量。
        temperature (float): 采样时的温度。
        max_new_tokens (int): 生成响应的最大长度。
    """
    model_path = f'/mnt/local/wxy/models/{model_name}'
    
    # 1. 使用 VLLM 加载模型
    # tensor_parallel_size 可以根据你的 GPU 数量设置，以实现多卡并行
    # trust_remote_code=True 对于很多模型是必需的
    print("Loading VLLM model...")
    llm = LLM(
        model=model_path
        , trust_remote_code=True
        , tensor_parallel_size=len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        , dtype=torch.float16
        , gpu_memory_utilization=0.8
        )
    print("Model loaded.")

    # 2. 准备所有 prompts
    prompts = []
    original_data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            question = data['question']
            # 组合 prompt，与原逻辑保持一致
            prompt = question + '\n' + get_prompts(data_name)
            prompts.append(prompt)
            original_data.append(data)

    # 3. 定义采样参数
    # 用于贪心搜索 (temperature=0)
    greedy_params = SamplingParams(
        n=1,
        temperature=0,
        max_tokens=max_new_tokens,
    )
    # 用于多样本采样
    sample_params = SamplingParams(
        n=num_samples,
        temperature=temperature,
        max_tokens=max_new_tokens,
        # 如果 temperature=0，VLLM 默认使用贪心搜索。
        # 如果 temperature > 0，则自动进行采样。
    )

    # 4. 批量生成响应
    # 第一次调用：为所有 prompts 生成贪心响应
    print(f"Generating greedy responses for {len(prompts)} prompts...")
    greedy_outputs = llm.generate(prompts, greedy_params)
    print("Greedy generation finished.")

    # 第二次调用：为所有 prompts 生成采样响应
    print(f"Generating {num_samples} samples for each of the {len(prompts)} prompts...")
    sample_outputs = llm.generate(prompts, sample_params)
    print("Sample generation finished.")

    # 5. 处理并保存结果
    data_responses_output_path = f'./datasets/{data_name}/responses'
    if not os.path.exists(data_responses_output_path):
        os.makedirs(data_responses_output_path)

    data_responses_output_file = f'{data_responses_output_path}/responses_vllm.json'

    with open(data_responses_output_file, 'w', encoding='utf-8') as w:
        for i in range(len(prompts)):
            # 获取原始数据
            result_data = original_data[i]
            
            # 提取贪心响应
            greedy_response = greedy_outputs[i].outputs[0].text.strip()
            
            # 提取采样响应
            sample_responses = [output.text.strip() for output in sample_outputs[i].outputs]
            
            # 组合最终结果
            result_data['greedy_response'] = greedy_response
            result_data['sample_responses'] = sample_responses
            
            # 写入文件
            w.write(json.dumps(result_data, ensure_ascii=False) + '\n')
            
    print(f"All responses have been saved to {data_responses_output_file}")
    return data_responses_output_file

def get_labels(model_name, data_name, dataset_path, num_samples=10, temperature=0.7, max_new_tokens=512):
    model_path = f'/mnt/local/wxy/models/{model_name}'
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompt = get_prompts(data_name) 
    data_responses_output_path = f'./datasets/{data_name}/responses'
    if not os.path.exists(data_responses_output_path):
        os.makedirs(data_responses_output_path)

    data_file_name = dataset_path.split('/')[-1].split('.')[0]
    data_responses_output_file = f'{data_responses_output_path}/responses_{data_file_name}.json'

    with open(dataset_path, 'r') as f, open(data_responses_output_file, 'w') as w:
        for line in f.readlines():
            data = json.loads(line.strip())
            question = data['question']
            prompt = question + '\n' + get_prompts(data_name)
            greedy_response = _generate(model=model, tokenizer=tokenizer, prompt=prompt, num_samples=1, temperature=0, max_new_tokens=max_new_tokens)[0]
            sample_responses = _generate(model=model, tokenizer=tokenizer, prompt=prompt, num_samples=num_samples, temperature=temperature, max_new_tokens=max_new_tokens)
            greedy_answer = gsm8k_answer_extractor(greedy_response)
            sample_answers = [gsm8k_answer_extractor(ans) for ans in sample_responses]

            new_data = {}
            new_data['question'] = question
            new_data['prompt'] = prompt
            new_data['greedy_response'] = greedy_response
            new_data['greedy_answer'] = greedy_answer
            new_data['sample_responses'] = sample_responses
            new_data['sample_answers'] = sample_answers
            new_data['label'] = get_label(greedy_answer, sample_answers)

            w.write(json.dumps(new_data, ensure_ascii=False)+'\n')

    return data_responses_output_file

def save_hidden_states(model_name, data_name, dataset_path):
    model_path = f'/mnt/local/wxy/models/{model_name}'

    print("正在加载模型和分词器...")
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


    data_responses_output_file = f'/mnt/local2/wxy/hallucination_predict_by_prompt/datasets/{data_name}/responses.json' 
    if not os.path.exists(data_responses_output_file):
        data_responses_output_file = get_labels(model_name, data_name, dataset_path)
        # data_responses_output_file = get_labels_vllm(model_name, data_name, dataset_path)

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

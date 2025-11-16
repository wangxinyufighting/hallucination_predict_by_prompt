from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch
import json
from tqdm import tqdm
import os
from vllm import LLM, SamplingParams
import re

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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, log_loss
)
from sklearn.model_selection import GridSearchCV
# 指定可见 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"


MMLU_PRO_PROMPT = """
Reason step by step about the correct answer based on the question and options provided. 
After your reasoning, you will select the most correct answer(e.g., A, B, C, D, F, G, H, I, J) and write it in \\boxed{}. 
For example: \\boxed{A}
Let's think step by step!
"""

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
        #print(f"Layer {layer_idx} attn_output shape: {attn_output.shape}")
        # 将 (batch, seq_len, hidden_size) -> (batch, seq_len, num_heads, head_dim)
        reshaped_output = attn_output.view(batch_size, seq_len, num_heads, head_dim)
        all_head_representations.append(reshaped_output[:, -1, :, :])
    
    all_head_representations = np.array(all_head_representations)  # (Layers, Heads, Head_Dim)

    all_layer_hidden_states = []
    for layer_idx, layer_hidden_state in layer_hidden_states.items():
        #print(f"Layer {layer_idx} layer_hidden_state shape: {layer_hidden_state.shape}")
        # 改:layer_hidden_states的shape这里只有两维
        # all_layer_hidden_states.append(layer_hidden_state[:, -1, :])
        all_layer_hidden_states.append(layer_hidden_state[-1, :])
    all_layer_hidden_states = np.array(all_layer_hidden_states)  # (Layers, Tokens, Hidden_Size)
    #all_layer_hidden_states = np.array(all_layer_hidden_states)

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

def get_prompts_and_answer_extractor(data_name):
    prompt = None
    answer_extractor = None

    if data_name.lower() == 'gsm8k':
        prompt = "Let's think step by step and output the final answer in \\boxed{}."
        answer_extractor = gsm8k_answer_extractor

    elif data_name.lower() == 'mmlupro':
        prompt = MMLU_PRO_PROMPT
        answer_extractor = mmlu_pro_answer_extractor

    return prompt, answer_extractor

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

def mmlu_pro_answer_extractor(solution_str, prompt=""):
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
    def is_number(x):
        try:
            float(x)
            return True
        except ValueError:
            return False
    # --- 分支处理 ---
    if is_number(greedy_answer) and is_number(majority_answer):
        # 数字型答案
        return float(greedy_answer) == float(majority_answer)
    else:
        # 非数字型（如 A/B/C/D）
        return greedy_answer.strip().upper() == majority_answer.strip().upper()
    #return float(greedy_answer) == float(majority_answer)


def _generate(model, tokenizer, prompt, max_new_tokens=2048, num_samples=1, temperature=0.7):
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

def get_labels_vllm(model_name, data_name, dataset_path, num_samples=10, temperature=0.7,max_new_tokens=2048):
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
        , gpu_memory_utilization=0.6
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
            prompt, answer_extractor = get_prompts_and_answer_extractor(data_name)
            prompt = question + '\n' + prompt
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

def get_labels(model_name, data_name, dataset_path, num_samples=10, temperature=0.7, max_new_tokens=2048):
    model_path = f'/mnt/local/wxy/models/{model_name}'
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompt, answer_extractor = get_prompts_and_answer_extractor(data_name) 
    data_responses_output_path = f'./datasets/{data_name}/responses'
    if not os.path.exists(data_responses_output_path):
        os.makedirs(data_responses_output_path)

    data_file_name = dataset_path.split('/')[-1].split('.')[0]
    data_responses_output_file = f'{data_responses_output_path}/responses_{data_file_name}.json'

    with open(dataset_path, 'r') as f, open(data_responses_output_file, 'w') as w:
        for line in f.readlines():
            data = json.loads(line.strip())
            question = data['question']
            prompt = question + '\n' + prompt
            greedy_response = _generate(model=model, tokenizer=tokenizer, prompt=prompt, num_samples=1, temperature=0, max_new_tokens=max_new_tokens)[0]
            sample_responses = _generate(model=model, tokenizer=tokenizer, prompt=prompt, num_samples=num_samples, temperature=temperature, max_new_tokens=max_new_tokens)
            greedy_answer = answer_extractor(greedy_response)
            sample_answers = [answer_extractor(ans) for ans in sample_responses]

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

def get_labels_and_hidden_vllm(model_name, data_name, dataset_path, num_samples=10, temperature=0.7, max_new_tokens=2048):
    """
    完整流程：
        1. 贪心生成 1 个响应
        2. 多样本采样 n 个响应
        3. 多数投票生成 label
        4. 获取 prompt 的 hidden states
        5. 保存 hidden states 与 label

    Returns:
        data_responses_output_file: 保存贪心/采样响应及 label 的 json 文件路径
    """
    model_path = f'/mnt/local/wxy/models/{model_name}'
    
    # --- 1. 加载 VLLM ---
    print("Loading VLLM model...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')),
        dtype="auto",
        gpu_memory_utilization=0.5
    )
    print("VLLM loaded.")

    # --- 2. 加载数据集 ---
    prompts, original_data = [], []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            question = data['question']
            prompt, answer_extractor = get_prompts_and_answer_extractor(data_name)
            prompt = question + '\n' + prompt
            prompts.append(prompt)
            original_data.append(data)
    
    # --- 3. 定义生成参数 ---
    greedy_params = SamplingParams(n=1, temperature=0, max_tokens=max_new_tokens)
    sample_params = SamplingParams(n=num_samples, temperature=temperature, max_tokens=max_new_tokens)

    # --- 4. 批量生成 ---
    print(f"Generating greedy responses for {len(prompts)} prompts...")
    greedy_outputs = llm.generate(prompts, greedy_params)
    print("Greedy generation finished.")

    print(f"Generating {num_samples} samples for each prompt...")
    sample_outputs = llm.generate(prompts, sample_params)
    print("Sample generation finished.")

    # --- 5. 处理 label 和 hidden states ---
    # 加载 HF 模型，用于获取 hidden states
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    all_hidden, all_label = [], []
    data_responses_output_path = f'./datasets/{data_name}/responses'
    os.makedirs(data_responses_output_path, exist_ok=True)
    data_responses_output_file = f'{data_responses_output_path}/responses_vllm.json'

    with open(data_responses_output_file, 'w', encoding='utf-8') as w:
        for i in tqdm(range(len(prompts))):
            prompt = prompts[i]
            result_data = original_data[i]

            # --- 贪心响应 ---
            greedy_resp = greedy_outputs[i].outputs[0].text.strip()
            
            # --- 多样本响应 ---
            sample_resps = [o.text.strip() for o in sample_outputs[i].outputs]

            # --- 提取答案 ---
            greedy_answer = answer_extractor(greedy_resp)
            sample_answers = [answer_extractor(ans) for ans in sample_resps]

            # --- 多数投票生成 label ---
            label = get_label(greedy_answer, sample_answers)

            # --- 获取 hidden states ---
            captured_data = get_and_save_representations(prompt, hf_model, tokenizer)
            hidden_states = captured_data["layer_hidden_states"]

            # --- 保存结果 ---
            all_hidden.append(hidden_states)
            all_label.append(label)

            result_data.update({
                "prompt": prompt,
                "greedy_response": greedy_resp,
                "greedy_answer": greedy_answer,
                "sample_responses": sample_resps,
                "sample_answers": sample_answers,
                "label": label
            })
            w.write(json.dumps(result_data, ensure_ascii=False) + '\n')

    # --- 保存 hidden states 和 labels ---
    path = './feature'
    os.makedirs(path, exist_ok=True)
    np.save(f'{path}/{data_name}_{model_name}_layers.npy', np.array(all_hidden))
    np.save(f'{path}/{data_name}_{model_name}_labels.npy', np.array(all_label))

    print(f"Hidden states and labels saved. JSON file: {data_responses_output_file}")
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


    data_responses_output_file = f'/home/zcy/hallucination_predict_by_prompt/datasets/{data_name}/responses.json' 
    if not os.path.exists(data_responses_output_file):
        # 改成 get_labels_vllm,通过采样获取标签
        data_responses_output_file = get_labels(model_name, data_name, dataset_path)
        # data_responses_output_file = get_labels_vllm(model_name, data_name, dataset_path)
        # data_responses_output_file = get_labels_and_hidden_vllm(model_name, data_name, dataset_path)

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

# def probe(layer_num, head_num, head_wise_activations_train, labels_train,
#           head_wise_activations_test, labels_test, max_iter=5, verbose=True):
    
#     # --- 存储指标 ---
#     m_train_auc = np.empty([layer_num, head_num], dtype=float)
#     m_train_auroc = np.empty([layer_num, head_num], dtype=float)
#     m_train_loss = np.empty([layer_num, head_num], dtype=float)

#     m_auc = np.empty([layer_num, head_num], dtype=float)
#     m_auroc = np.empty([layer_num, head_num], dtype=float)

#     d_clf = {}

#     for layer in tqdm(range(layer_num), desc="Probing layers"):
#         for head in range(head_num):
#             # --- 训练集 ---
#             X_train = head_wise_activations_train[:, layer, head, :]
#             Y_train = labels_train
#             assert X_train.shape[0] == Y_train.shape[0]

#             clf = LogisticRegression(max_iter=max_iter).fit(X_train, Y_train)

#             train_logits = clf.predict_proba(X_train)[:, 1]
#             train_preds = clf.predict(X_train)

#             precision_list, recall_list, _ = precision_recall_curve(Y_train, train_logits)
#             train_prauc = auc(recall_list, precision_list)
#             train_auroc = roc_auc_score(Y_train, train_logits)
#             train_loss = log_loss(Y_train, train_logits)

#             m_train_auc[layer][head] = train_prauc
#             m_train_auroc[layer][head] = train_auroc
#             m_train_loss[layer][head] = train_loss

#             # --- 测试集 ---
#             X_test = head_wise_activations_test[:, layer, head, :]
#             Y_test = labels_test
#             assert X_test.shape[0] == Y_test.shape[0]

#             if layer not in d_clf:
#                 d_clf[layer] = {}
#             d_clf[layer][head] = clf

#             test_logits = clf.predict_proba(X_test)[:, 1]
#             test_preds = clf.predict(X_test)

#             precision_list, recall_list, _ = precision_recall_curve(Y_test, test_logits)
#             prauc_test = auc(recall_list, precision_list)
#             auroc_test = roc_auc_score(Y_test, test_logits)

#             m_auc[layer][head] = prauc_test
#             m_auroc[layer][head] = auroc_test

#             # --- 打印每层每 head ---
#             if verbose:
#                 print(
#                     f"Layer {layer:2d}, Head {head:2d} | "
#                     f"Train PR-AUC={train_prauc:.3f}, ROC-AUC={train_auroc:.3f}, Loss={train_loss:.3f} | "
#                     f"Test PR-AUC={prauc_test:.3f}, ROC-AUC={auroc_test:.3f}"
#                 )

#     # --- 返回结果 ---
#     return  m_auc, m_auroc, d_clf
# 构造测试集
def extract_gt_answer(answer_text: str) -> str:
    """
    从 test.json 的 'answer' 字段中提取标准答案。
    示例："He ... #### 540" -> "540"
    """
    match = re.search(r'####\s*([\d\.\-]+)', answer_text)
    if match:
        return match.group(1).strip()
    # 匹配单个字母选项（A-J）
    match = re.match(r'^\s*([A-Ja-j])\s*$', answer_text)
    if match:
        return match.group(1).strip().upper()
    return ""  # 没找到返回空字符串
    
def generate_and_save_features(model_name, data_name, dataset_path, split="train",
                                  num_samples=10, temperature=0.7, max_new_tokens=2048):
    """
    统一处理 train/test 数据（不依赖 VLLM）：
      - train: 贪心 + 多样本采样 → 多数投票标签
      - test: 贪心 + 多样本采样 → 与标准答案比对标签
    最后保存 hidden states / attention / labels

    Args:
        model_name: 模型名称
        data_name: 数据集名称
        dataset_path: 数据集 jsonl 路径
        split: "train" 或 "test"
        num_samples: 每个 prompt 的采样数量
        temperature: 采样温度
        max_new_tokens: 最大生成 token 数
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    import torch

    model_path = f'/mnt/local/wxy/models/{model_name}'
    print(f"加载模型 {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    responses_dir = f'./datasets/{data_name}/responses'
    os.makedirs(responses_dir, exist_ok=True)
    feature_dir = './feature'
    os.makedirs(feature_dir, exist_ok=True)

    # --- 准备 prompts 和原始数据 ---
    prompts, original_data = [], []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            question = data['question']
            prompt, answer_extractor = get_prompts_and_answer_extractor(data_name)
            prompt = question + '\n' + prompt
            prompts.append(prompt)
            original_data.append(data)

    # --- 批量生成函数 ---
    def generate_responses(prompt_list, num_samples, temperature):
        outputs_all = []
        for prompt in tqdm(prompt_list):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

            # 贪心生成
            greedy_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
            greedy_text = tokenizer.decode(greedy_ids[0], skip_special_tokens=True)

            # 采样生成
            sampled_texts = []
            for _ in range(num_samples):
                sample_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature
                )
                sampled_texts.append(tokenizer.decode(sample_ids[0], skip_special_tokens=True))

            outputs_all.append((greedy_text, sampled_texts))
        return outputs_all

    print(f"生成响应 ({split} 集)...")
    all_outputs = generate_responses(prompts, num_samples, temperature)

    # --- 保存 JSON & 提取特征 ---
    data_responses_output_file = f'{responses_dir}/responses_{split}.json'

    all_hidden, all_attention, all_label = [], [], []
    for i, (greedy_resp, sample_resps) in enumerate(tqdm(all_outputs)):
        prompt = prompts[i]
        result_data = original_data[i]

        greedy_answer = answer_extractor(greedy_resp)
        sample_answers = [answer_extractor(ans) for ans in sample_resps]

        # --- label 逻辑 ---
        if split == "train":
            label = get_label(greedy_answer, sample_answers)
        else:
            gt_answer_raw = result_data['answer']
            gt_answer = extract_gt_answer(gt_answer_raw)
            try:
                label = (float(greedy_answer) == float(gt_answer))
            except:
                label = (greedy_answer.strip() == gt_answer.strip())
            result_data['gt_answer'] = gt_answer

        # --- 获取 hidden states / attention ---
        captured_data = get_and_save_representations(prompt, model, tokenizer)
        hidden_states = captured_data["layer_hidden_states"]
        attention_heads = captured_data["all_head_representations"]

        all_hidden.append(hidden_states)
        all_attention.append(attention_heads)
        all_label.append(label)

        # --- 更新 JSON 数据 ---
        result_data.update({
            "prompt": prompt,
            "greedy_response": greedy_resp,
            "greedy_answer": greedy_answer,
            "sample_responses": sample_resps,
            "sample_answers": sample_answers,
            "label": label
        })
        with open(data_responses_output_file, 'a', encoding='utf-8') as w:
            w.write(json.dumps(result_data, ensure_ascii=False) + '\n')

    # --- 保存 .npy 特征 ---
    np.save(f'{feature_dir}/{data_name}_{model_name}_{split}_layers.npy', all_hidden)
    np.save(f'{feature_dir}/{data_name}_{model_name}_{split}_heads.npy', all_attention)
    np.save(f'{feature_dir}/{data_name}_{model_name}_{split}_labels.npy', all_label)

    print(f"{split} 数据集处理完成，特征已保存：")
    print(f" → {feature_dir}/{data_name}_{model_name}_{split}_layers.npy")
    print(f" → {feature_dir}/{data_name}_{model_name}_{split}_heads.npy")
    print(f" → {feature_dir}/{data_name}_{model_name}_{split}_labels.npy")

    return data_responses_output_file

def generate_and_save_features_vllm(model_name, data_name, dataset_path, split="train",
                               num_samples=10, temperature=0.7, max_new_tokens=2048):
    """
    统一处理 train/test 数据：
      - train: 贪心 + 多样本采样 → 多数投票标签
      - test: 贪心  → 与标准答案比对标签
    最后保存 hidden states / attention / labels
    
    Args:
        model_name: 模型名称
        data_name: 数据集名称
        dataset_path: 数据集 jsonl 路径
        split: "train" 或 "test"
        num_samples: 每个 prompt 的采样数量
        temperature: 采样温度
        max_new_tokens: 最大生成 token 数
    """
    model_path = f'/mnt/local/wxy/models/{model_name}'

    print(f"加载模型 {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    responses_dir = f'./datasets/{data_name}/responses'
    os.makedirs(responses_dir, exist_ok=True)
    feature_dir = './feature'
    os.makedirs(feature_dir, exist_ok=True)

    # --- 准备 prompts 和原始数据 ---
    prompts, original_data = [], []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            question = data['question']
            prompt, answer_extractor = get_prompts_and_answer_extractor(data_name)
            prompt = question + '\n' + prompt
            prompts.append(prompt)
            original_data.append(data)

    # --- 使用 VLLM 生成贪心与采样响应 ---
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')),
        dtype="auto",
        gpu_memory_utilization=0.5
    )

    greedy_params = SamplingParams(n=1, temperature=0, max_tokens=max_new_tokens)
    sample_params = SamplingParams(n=num_samples, temperature=temperature, max_tokens=max_new_tokens)

    print(f"生成贪心响应 ({len(prompts)} prompts)...")
    greedy_outputs = llm.generate(prompts, greedy_params)
    print("贪心生成完成。")
    print(f"生成 {num_samples} 采样响应...")
    sample_outputs = llm.generate(prompts, sample_params)
    print("采样生成完成。")

    # --- 保存 JSON 响应 & label ---
    data_responses_output_file = f'{responses_dir}/responses_{split}.json'
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16)

    all_hidden, all_attention, all_label = [], [], []

    with open(data_responses_output_file, 'w', encoding='utf-8') as w:
        for i in tqdm(range(len(prompts))):
            prompt = prompts[i]
            result_data = original_data[i]

            greedy_resp = greedy_outputs[i].outputs[0].text.strip()
            sample_resps = [o.text.strip() for o in sample_outputs[i].outputs]

            greedy_answer = answer_extractor(greedy_resp)
            sample_answers = [answer_extractor(ans) for ans in sample_resps]
            # --- label 逻辑 ---
            if split == "train":
                # 多数投票
                label = get_label(greedy_answer, sample_answers)
            else:
                # 测试集：对比标准答案
                gt_answer_raw = result_data['answer']
                gt_answer = extract_gt_answer(gt_answer_raw)
                def is_number(x):
                    try:
                        float(x)
                        return True
                    except ValueError:
                        return False
                # --- 分支处理 ---
                if is_number(greedy_answer) and is_number(gt_answer):
                    # 数字型答案
                    label = float(greedy_answer) == float(gt_answer)
                else:
                    # 非数字型（如 A/B/C/D）
                    label = greedy_answer.strip().upper() == gt_answer.strip().upper()              
                # try:
                #     label = (float(greedy_answer) == float(gt_answer))
                # except:
                #     label = (greedy_answer.strip() == gt_answer.strip())
                result_data['gt_answer'] = gt_answer

            # --- 获取 hidden states ---
            captured_data = get_and_save_representations(prompt, hf_model, tokenizer)
            hidden_states = captured_data["layer_hidden_states"]
            attention_heads = captured_data["all_head_representations"]

            all_hidden.append(hidden_states)
            all_attention.append(attention_heads)
            all_label.append(label)

            # --- 更新 JSON 数据并写入 ---
            result_data.update({
                "prompt": prompt,
                "greedy_response": greedy_resp,
                "greedy_answer": greedy_answer,
                "sample_responses": sample_resps,
                "sample_answers": sample_answers,
                "label": label
            })
            w.write(json.dumps(result_data, ensure_ascii=False) + '\n')

    # --- 保存 .npy 特征 ---
    np.save(f'{feature_dir}/{data_name}_{model_name}_{split}_layers.npy', all_hidden)
    np.save(f'{feature_dir}/{data_name}_{model_name}_{split}_heads.npy', all_attention)
    np.save(f'{feature_dir}/{data_name}_{model_name}_{split}_labels.npy', all_label)

    print(f"{split} 数据集处理完成，特征已保存：")
    print(f" → {feature_dir}/{data_name}_{model_name}_{split}_layers.npy")
    print(f" → {feature_dir}/{data_name}_{model_name}_{split}_heads.npy")
    print(f" → {feature_dir}/{data_name}_{model_name}_{split}_labels.npy")
    return data_responses_output_file



def generate_and_save_features_vllm_baseline(model_name, data_name, dataset_path, split="train", max_new_tokens=2048):
    """
    baseline 版本：
      - train/test: 都是 仅贪心生成 → 与标准答案比对标签
      - 文件名带 `_baseline`
    """
    import os, json, torch, re
    import numpy as np
    from tqdm import tqdm
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_path = f'/mnt/local/wxy/models/{model_name}'
    print(f"加载模型 {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    responses_dir = f'./datasets/{data_name}/responses'
    os.makedirs(responses_dir, exist_ok=True)
    feature_dir = './feature'
    os.makedirs(feature_dir, exist_ok=True)

    # --- 准备 prompts 和原始数据 ---
    prompts, original_data = [], []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            question = data['question']
            prompt,answer_extractor = get_prompts_and_answer_extractor(data_name)
            prompt = question + '\n' + prompt
            prompts.append(prompt)
            original_data.append(data)

    # --- 使用 VLLM 生成贪心响应 ---
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')),
        dtype="auto",
        gpu_memory_utilization=0.5
    )

    greedy_params = SamplingParams(n=1, temperature=0, max_tokens=max_new_tokens)

    print(f"生成贪心响应 ({len(prompts)} prompts)...")
    greedy_outputs = llm.generate(prompts, greedy_params)
    print("贪心生成完成。")

    # --- 保存 JSON 响应 & label ---
    data_responses_output_file = f'{responses_dir}/responses_{split}_baseline.json'
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16)

    all_hidden, all_attention, all_label = [], [], []

    with open(data_responses_output_file, 'w', encoding='utf-8') as w:
        for i in tqdm(range(len(prompts))):
            prompt = prompts[i]
            result_data = original_data[i]

            greedy_resp = greedy_outputs[i].outputs[0].text.strip()
            greedy_answer = answer_extractor(greedy_resp)

            # --- 提取标准答案并生成布尔标签 ---
            gt_answer_raw = result_data['answer']
            gt_answer = extract_gt_answer(gt_answer_raw)
            def is_number(x):
                try:
                    float(x)
                    return True
                except ValueError:
                    return False
            # --- 分支处理 ---
            if is_number(greedy_answer) and is_number(gt_answer):
                # 数字型答案
                label = float(greedy_answer) == float(gt_answer)
            else:
                # 非数字型（如 A/B/C/D）
                label = greedy_answer.strip().upper() == gt_answer.strip().upper()   
            # try:
            #     label = (float(greedy_answer) == float(gt_answer))
            # except:
            #     label = (greedy_answer.strip() == gt_answer.strip())
            result_data['gt_answer'] = gt_answer

            # --- 获取 hidden states ---
            captured_data = get_and_save_representations(prompt, hf_model, tokenizer)
            hidden_states = captured_data["layer_hidden_states"]
            attention_heads = captured_data["all_head_representations"]

            all_hidden.append(hidden_states)
            all_attention.append(attention_heads)
            all_label.append(label)

            # --- 写入 JSON ---
            result_data.update({
                "prompt": prompt,
                "greedy_response": greedy_resp,
                "greedy_answer": greedy_answer,
                "label": label
            })
            w.write(json.dumps(result_data, ensure_ascii=False) + '\n')

    # --- 保存 .npy 特征 ---
    np.save(f'{feature_dir}/{data_name}_{model_name}_{split}_baseline_layers.npy', all_hidden)
    np.save(f'{feature_dir}/{data_name}_{model_name}_{split}_baseline_heads.npy', all_attention)
    np.save(f'{feature_dir}/{data_name}_{model_name}_{split}_baseline_labels.npy', all_label)

    print(f"{split} (baseline) 数据集处理完成，特征已保存：")
    print(f" → {feature_dir}/{data_name}_{model_name}_{split}_baseline_layers.npy")
    print(f" → {feature_dir}/{data_name}_{model_name}_{split}_baseline_heads.npy")
    print(f" → {feature_dir}/{data_name}_{model_name}_{split}_baseline_labels.npy")
    return data_responses_output_file

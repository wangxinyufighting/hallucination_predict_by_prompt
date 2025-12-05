import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ql_uncertainty import QLUncertainty
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
# ---------- 提取 mmlu 最终答案 ----------
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
def compute_prr(labels, probs):
    probs = np.array(probs, dtype=float)
    labels = np.array(labels)

    # uncertainty
    uncertainty = 1 - np.maximum(probs, 1 - probs)

    # 排序
    sorted_idx = np.argsort(uncertainty)
    sorted_labels = labels[sorted_idx]

    total = len(labels)
    cumulative_correct = np.cumsum(sorted_labels)

    # coverage (1..N)/N
    coverage = np.arange(1, total + 1) / total

    # risk = 1 - accuracy@coverage
    risk = 1 - cumulative_correct / np.arange(1, total + 1)

    # AURC
    aurc = np.trapz(risk, coverage)

    # ---- 计算 AURC_optimal ----
    num_errors = total - np.sum(labels)
    optimal_risk = np.concatenate([
        np.ones(num_errors),         # 错误全部放在前面
        np.zeros(total - num_errors) # 正确全部在后
    ])
    aurc_optimal = np.trapz(optimal_risk, coverage)

    # ---- 计算 AURC_random ----
    error_rate = num_errors / total
    aurc_random = error_rate * (1 - error_rate / 2)

    # ---- 正确PRR公式 ----
    prr = (aurc - aurc_optimal) / (aurc_random - aurc_optimal)

    return prr, aurc, aurc_random

# ---------- 配置 ----------
model_name = '/mnt/local/wxy/models/Qwen2.5-3B-Instruct'
data_file = '/mnt/local2/zcy/hallucination_predict_by_prompt/datasets/mmlupro/test-500.json'

# ---------- 加载模型 ----------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map={"": "cuda:0"}
)
model.set_attn_implementation("eager")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

# ---------- 读取 mmlu ----------
examples = []
with open(data_file, "r", encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line.strip()))

prompt_suffix = """
Reason step by step about the correct answer based on the question and options provided. 
After your reasoning, you will select the most correct answer(e.g., A, B, C, D, F, G, H, I, J) and write it in \\boxed{}. 
For example: \\boxed{A}
Let's think step by step!
"""


labels = []
scores = []
responses = [] 
results = []
# ---------- 循环处理每条数据，带进度条 ----------
for ex in tqdm(examples, desc="Processing mmlu", ncols=100):
    question = ex['question']
    gt_answer = ex['answer']
    prompt = question + "\n" + prompt_suffix

    # --- 模型 greedy 推理 ---
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
        )
    pred_text = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    responses.append(pred_text)
    # --- 提取预测答案中的最终答案 ---

    pred_answer = mmlu_pro_answer_extractor(pred_text)

    print("pred_answer:",pred_answer)
    # --- 生成标签 0/1 ---

    label = pred_answer.upper() == gt_answer.upper()
    labels.append(int(label))
    # 根据 pred_answer 生成 target_tokens
    target_tokens = []
    for ch in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        ids = tokenizer(ch, add_special_tokens=False).input_ids
        target_tokens.append(ids)
    #answer_token_ids = tokenizer(str(pred_answer), add_special_tokens=False).input_ids
    qlu = QLUncertainty(
        model,
        tokenizer,
        method='internal_confidence',
        target_tokens=[target_tokens]
    )
    # --- 计算 QLUncertainty 分数 ---
    score = qlu.estimate(prompt)
    scores.append(score)
    # --- 实时输出当前 label 和 score ---
    tqdm.write(f"label={label}, score={score:.4f}, response={pred_text}") 
    results.append({
    "question": question,
    "prompt": prompt,
    "gt_answer": gt_answer,
    "pred_answer": pred_answer,
    "response": pred_text,
    "label": int(label),
    "score": score
})
    
# ---------- 保存 JSONL 文件 ----------
with open('mmlu_test-500_results.jsonl', "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
labels = []
scores = []
# 读取 jsonl
with open('mmlu_test-500_results.jsonl', "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line.strip())
        labels.append(r["label"])
        scores.append(r["score"])  # 模型置信分数        
# ---------- 计算 ROC AUC ----------
precision, recall, _ = precision_recall_curve(labels, scores)
prauc = auc(recall, precision)
auc = roc_auc_score(labels, scores)
prr, aurc, aurc_random = compute_prr(labels, scores)
print("mmlu test AUC:", auc)
print("mmlu test prauc:", prauc)
print("mmlu test PRR:", prr)
# 要写入 JSON 的内容
result_dict = {
    "auc": auc,
    "prauc": prauc,
    "prr": prr,
    "aurc": aurc,
    "aurc_random": aurc_random
}

# 保存到 json 文件
with open("mmlu_test-500_metrics.json", "w", encoding="utf-8") as f:
    json.dump(result_dict, f, ensure_ascii=False, indent=4)

print("指标已写入 mmlu_test_metrics.json")

import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ql_uncertainty import QLUncertainty
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
# ---------- 提取 GSM8K 最终答案 ----------
def extract_gt_answer(answer_text):
    match = re.search(r"####\s*(.*)", answer_text)
    if match:
        return match.group(1).strip()
    else:
        return answer_text.strip()
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
data_file = '/mnt/local2/zcy/hallucination_predict_by_prompt/datasets/gsm8k/train_30.jsonl'

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

# ---------- 读取 GSM8K ----------
examples = []
with open(data_file, "r", encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line.strip()))

prompt_suffix = "Let's think step by step and output the final answer in \\boxed{}."


labels = []
scores = []
responses = [] 
results = []
# ---------- 循环处理每条数据，带进度条 ----------
for ex in tqdm(examples, desc="Processing GSM8K", ncols=100):
    question = ex['question']
    gt_answer = extract_gt_answer(ex['answer'])
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
    boxed_match = re.search(r"\\boxed\{(.*?)\}", pred_text)
    if boxed_match:
        pred_answer = boxed_match.group(1).strip()
    else:
        # 如果没找到 boxed，
        pred_answer = -1000000
    print("pred_answer:",pred_answer)
    # --- 生成标签 0/1 ---
    try:
        label = float(pred_answer) == float(gt_answer)
    except:
        label = pred_answer.upper() == gt_answer.upper()
    labels.append(int(label))
    # 根据 pred_answer 生成 target_tokens
    answer_token_ids = tokenizer(str(pred_answer), add_special_tokens=False).input_ids
    qlu = QLUncertainty(
        model,
        tokenizer,
        method='internal_confidence',
        target_tokens=[answer_token_ids]
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
with open('gsm8k_train_results.jsonl', "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
labels = []
scores = []
# 读取 jsonl
with open('gsm8k_test_results.jsonl', "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line.strip())
        labels.append(r["label"])
        scores.append(r["score"])  # 模型置信分数        
# ---------- 计算 ROC AUC ----------
precision, recall, _ = precision_recall_curve(labels, scores)
prauc = auc(recall, precision)
auc = roc_auc_score(labels, scores)
prr, aurc, aurc_random = compute_prr(labels, scores)
print("GSM8K test AUC:", auc)
print("GSM8K test prauc:", prauc)
print("GSM8K test PRR:", prr)
# 要写入 JSON 的内容
result_dict = {
    "auc": auc,
    "prauc": prauc,
    "prr": prr,
    "aurc": aurc,
    "aurc_random": aurc_random
}

# 保存到 json 文件
with open("gsm8k_test_metrics.json", "w", encoding="utf-8") as f:
    json.dump(result_dict, f, ensure_ascii=False, indent=4)

print("指标已写入 gsm8k_test_metrics.json")

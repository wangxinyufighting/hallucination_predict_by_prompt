from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json, os

model_path = "/mnt/local/wxy/models/Qwen2.5-3B-Instruct"

llm = LLM(
    model=model_path,
    trust_remote_code=True,
    tensor_parallel_size=1,
    dtype="auto",
    gpu_memory_utilization=0.3
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# ✅ 测试 prompt
prompt = "John has 3 apples and buys 5 more. How many apples does he have now?"

# ✅ greedy 参数（确定性）
greedy_params = SamplingParams(
    n=1,
    temperature=0,
    max_tokens=128,
)

def run_once(prompt):
    out = llm.generate([prompt], greedy_params)[0].outputs[0].text.strip()
    return out

def extract_answer(text):
    # 你自己的 gsm8k parser
    # 这里简单示例，你可以替换
    import re
    match = re.search(r"(\d+)", text)
    return match.group(1) if match else text

# ✅ 实验 1：单条生成两次
out1 = run_once(prompt)
out2 = run_once(prompt)

print("=== 单条生成 ===")
print("response equal? ", out1 == out2)
print("answer equal?   ", extract_answer(out1) == extract_answer(out2))
print("out1:", out1)
print("out2:", out2)

# ✅ 实验 2：多条 batch 生成
batch_prompts = [prompt, prompt]
outs1 = llm.generate(batch_prompts, greedy_params)
outs2 = llm.generate(batch_prompts, greedy_params)

batch1 = [o.outputs[0].text.strip() for o in outs1]
batch2 = [o.outputs[0].text.strip() for o in outs2]

print("\n=== 批量生成（batch=2）===")
print("batch equal? ", batch1 == batch2)
print("answers equal? ", 
      [extract_answer(x) for x in batch1] ==
      [extract_answer(x) for x in batch2])

print("batch1:", batch1)
print("batch2:", batch2)

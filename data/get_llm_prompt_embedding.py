from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel
import numpy as np
from transformers.generation.utils import GenerationConfig
import torch
import json
from tqdm import tqdm
import os

class LM_nnsight():
    def __init__(self, model_path, device="cuda", temperature=0.):
        self.device = device
        base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='sequential', torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model.generation_config = GenerationConfig.from_pretrained(model_path)
        if temperature == 0:
            base_model.generation_config.do_sample = False
            base_model.generation_config.temperature = None
            base_model.generation_config.top_p = None
            base_model.generation_config.top_k = None
        else:
            base_model.generation_config.do_sample = True
            base_model.generation_config.temperature = temperature

        # base_model = base_model.to(torch.float16)
        # base_model.to(self.device)
        base_model.eval()
        self.model = LanguageModel(base_model, tokenizer=tokenizer)

    def get_all_states(self, prompt):
        n_layers = len(self.model.model.layers)
        n_heads = self.model.model.config.num_attention_heads
        head_dim = int(self.model.model.config.hidden_size / n_heads)
        
        all_hidden_states = []
        all_attention_states = []
        with self.model.invoke(prompt) as invoker:
            for layer in self.model.model.layers:
                all_attention_states.append(layer.self_attn.output[0].save())
                all_hidden_states.append(layer.output[0].save())
        
        all_hidden_states_numpy = []
        all_attention_states_numpy = []
        for HS, AS in zip(all_hidden_states, all_attention_states):
            all_hidden_states_numpy.append(HS.value[0].cpu().numpy())
            atts = AS.value[0].cpu().numpy()
            all_attention_states_numpy.append(atts.reshape(atts.shape[0], n_heads, -1))
        all_hidden_states_numpy = np.array(all_hidden_states_numpy)
        all_attention_states_numpy = np.array(all_attention_states_numpy)
        
        return all_hidden_states_numpy, all_attention_states_numpy
        # all_hidden_states: (Layers, Tokens, 4096)
        # all_attention_states: (Layers, Tokens, Heads, 128)


model_name = 'vicuna-7b'
# model_name = 'Mistral-7B-Instruct-v0.2'
# model_name = 'llama2-7b-chat-hf'
# model_name = 'vicuna-13b'
# model_name = 'vicuna-33b'


model_path = f'/home/wxy/models/{model_name}'

llm = LM_nnsight(model_path=model_path, device='cuda')

all_hidden = []
all_attention = []
all_label = []

data_name = 'gsm8k'

file_path = f'../data/datasets/gsm8k/0shot/output/0shot/train_0_1000_{model_name}_1_sample_has_label.json'
with open(file_path, 'r') as f:
    for line in tqdm(f.readlines()):
        data = json.loads(line.strip())
        prompt = data['prompt']
        hidden_states_numpy, attention_states_numpy = llm.get_all_states(prompt)

        all_hidden.append(hidden_states_numpy[:,-1,:])
        all_attention.append(attention_states_numpy[:,-1,:,:])
        all_label.append(data['label'])


np.save(f'./feature/{data_name}_{model_name}_layers.npy', all_hidden)
np.save(f'./feature/{data_name}_{model_name}_heads.npy', all_attention)
np.save(f'./feature/{data_name}_{model_name}_labels.npy', all_label)

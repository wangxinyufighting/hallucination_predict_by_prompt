from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel
import numpy as np
from transformers.generation.utils import GenerationConfig
import json
import torch
from tqdm import tqdm


class LM_nnsight():
    def __init__(self, model_path, device="cuda", temperature=0.):
        self.device = device
        base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # if tokenizer.pad_token is None:
        #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        #     base_model.resize_token_embeddings(len(tokenizer))

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model.generation_config = GenerationConfig.from_pretrained(model_path)
        if temperature == 0:
            base_model.generation_config.do_sample = False
            base_model.generation_config.temperature = None
            base_model.generation_config.top_p = None
            base_model.generation_config.top_k = None
            #print(base_model.generation_config.temperature)
        else:
            base_model.generation_config.temperature = temperature
        base_model.to(self.device)
        base_model.eval()
        self.model = LanguageModel(base_model, tokenizer=tokenizer)
        
    # def generate_response(self, prompt, max_new_tokens=2000):
    #     with self.model.generate(max_new_tokens=max_new_tokens) as generator:  
    #         with generator.invoke(prompt) as invoker:
    #             pass
    #     return self.model.tokenizer.decode(generator.output[0])
    
    def generate_response(self, prompt, max_new_tokens=2000):
        with self.model.generate(max_new_tokens=max_new_tokens) as generator:  
            with generator.invoke(prompt) as invoker:
                pass
        return self.model.tokenizer.decode(generator.output[0])
                
    def __call__(self, prompt, max_new_tokens=2000):
        ans = self.generate_response(prompt, max_new_tokens)
        return ans

    def get_all_states(self, prompt):
        n_layers = len(self.model.model.layers)
        n_heads = self.model.model.config.num_attention_heads
        head_dim = int(self.model.model.config.hidden_size / n_heads)
        
        all_hidden_states = []
        all_attention_states = []
        with self.model.invoke(prompt) as invoker:
            for layer in self.model.model.layers:
                # print(layer.output[0].save().shape)
                all_attention_states.append(layer.self_attn.head_out[0].save())
                all_hidden_states.append(layer.output[0].save())
        
        all_hidden_states_numpy = []
        all_attention_states_numpy = []
        for HS, AS in zip(all_hidden_states, all_attention_states):
            all_hidden_states_numpy.append(HS.value[0].cpu().numpy())
            atts = AS.value[0].cpu().numpy()
            all_attention_states_numpy.append(atts)
            # all_attention_states_numpy.append(atts.reshape(atts.shape[0], n_heads, -1))
        all_hidden_states_numpy = np.array(all_hidden_states_numpy)
        all_attention_states_numpy = np.array(all_attention_states_numpy)
        
        return all_hidden_states_numpy, all_attention_states_numpy
        # all_hidden_states: (Layers, Tokens, 4096)
        # all_attention_states: (Layers, Tokens, Heads, 128)

    def intervention(self, prompt, interventions_dict, alpha=10, max_new_tokens=3):
        n_layers = len(self.model.model.layers)
        n_heads = self.model.model.config.num_attention_heads
        head_dim = int(self.model.model.config.hidden_size / n_heads)
        with self.model.generate(max_new_tokens=max_new_tokens) as generator:
            with generator.invoke(prompt) as invoker:
                for idx in range(max_new_tokens):
                    for layer_id, layer in enumerate(self.model.model.layers):
                        if layer_id in interventions_dict:
                            for (head, dir, std, _) in interventions_dict[layer_id]:
                                layer.self_attn.output[0][0, -1, head * head_dim: (head + 1) * head_dim] += alpha * std * dir
                    invoker.next()
        return self.model.tokenizer.decode(generator.output[0])



model_name = 'llama2-7b-chat-hf'
# model_name = 'Mistral-7B-Instruct-v0.2'

model_path = f'/mnt/local/xywang/models/{model_name}'
# base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
# base_model = base_model.to(torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# base_model.to('cuda')
# base_model.eval()
llm_1 = LM_nnsight(model_path=model_path, device="cuda")
# llm_2 = LM_nnsight(model_path=model_path_2, device="cpu")

data_path = '/home/wxy/project/hallucination_predict_by_prompt/data/datasets/gsm8k/examples.json'

l1 = []
a1 = []

with open(data_path, 'r') as f:
    for line in tqdm(f.readlines()):
        data = json.loads(line.strip())
        question = data['question']
        # prompt = f'You are a helpful assistant. Please answer the question. \n\n QUESTION:{question} \n\n ANSWER: '
        l_1,a_1 = llm_1.get_all_states(prompt=question)
        print(a_1)
        
        
# np.save(f'l_{model_name}.npy', l1)


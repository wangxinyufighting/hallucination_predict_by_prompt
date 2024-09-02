import datasets
import random
from rouge import Rouge
import pickle
import config
import os

SEED = 10
random.seed(SEED)

rouge = Rouge()


def get_label(response, answers):
    if isinstance(answers, list):
        for answer in answers:
            scores = rouge.get_scores(hypothesis=response, reference=answer)
            rougeL = scores[0]['rouge-l']['f']
            if rougeL > 0.3:
                return 0
            
    elif isinstance(answers, str):
        scores = rouge.get_scores(hypothesis=response, reference=answers)
        rougeL = scores[0]['rouge-l']['f']
        if rougeL > 0.3:
            return 0
        
    return 1


def get_trivia_qa_type(type, num_data):
    if type not in ["validation", "train"]:
        print(f'error! invalid type: {type}.')
        return None

    data = datasets.load_dataset('trivia_qa', "rc.nocontext", split=type) 
    data_samples = data.select(random.sample(list(range(data.num_rows)), num_data)) 

    path = f'{config.output_dir}/trivia_qa/{type}'
    if not os.path.exists(path):
        os.makedirs(path)

    data_samples.save_to_disk(path)


def get_trivia_qa(num_test, num_train):
    get_trivia_qa_type('train', num_train)
    get_trivia_qa_type('validation', num_test)


def get_squad_type(type, num_data):
    if type not in ["validation", "train"]:
        print(f'error! invalid type: {type}.')
        return None

    data = datasets.load_dataset('rajpurkar/squad', split=type) 
    data_samples = data.select(random.sample(list(range(data.num_rows)), num_data)) 

    path = f'{config.output_dir}/squad/{type}'
    if not os.path.exists(path):
        os.makedirs(path)

    data_samples.save_to_disk(path)

def get_squad(num_test, num_train):
    get_squad_type('train', num_train)
    get_squad_type('validation', num_test)


if __name__ == '__main__':
    # get_trivia_qa(300, 1500)
    get_squad(300, 1500)
    
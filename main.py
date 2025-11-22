from utils import *
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen2.5-3B-Instruct", help="Model name.")
    parser.add_argument("--data_name", type=str, default='gsm8k', help="dataset name, like 'gsm8k'")
    parser.add_argument("--dataset_path", type=str, default='./datasets/gsm8k/train_30.jsonl', help="Path to the dataset file.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens for sampling.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of sampled responses per prompt.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    save_hidden_states(args)

    # heads_file = '/mnt/local2/wxy/hallucination_predict_by_prompt/feature/gsm8k_Qwen2.5-3B-Instruct_heads.npy'
    # layer_file = '/mnt/local2/wxy/hallucination_predict_by_prompt/feature/gsm8k_Qwen2.5-3B-Instruct_layers.npy'
    # labels_file = '/mnt/local2/wxy/hallucination_predict_by_prompt/feature/gsm8k_Qwen2.5-3B-Instruct_labels.npy'

    # head_activations = np.load(heads_file)
    # # layer_wise_activations = np.load(layer_file)
    # labels = np.load(labels_file)

    # head_activations = head_activations[:, :, 0, :, :]

    # head_wise_activations_train = head_activations[:20, ...]
    # labels_train = labels[:20]
    # head_wise_activations_test = head_activations[10:, ...]
    # labels_test = labels[10:]

    # print(head_wise_activations_train.shape)

    # layer_num = head_wise_activations_train.shape[1]
    # head_num = head_wise_activations_train.shape[2]

    # print('layer_num:', layer_num)
    # print('head_num:', head_num)

    # m_auc, m_auroc, d_clf = probe(layer_num, head_num, head_wise_activations_train, labels_train, head_wise_activations_test, labels_test, max_iter=5)

    # print(m_auc)
    # print("m_auc:", m_auc.max())
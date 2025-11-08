from utils import *
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen2.5-3B-Instruct", help="Model name.")
    parser.add_argument("--data_name", type=str, default='gsm8k', help="dataset name, like 'gsm8k'")
    parser.add_argument("--train_dataset_path", type=str, default='./datasets/gsm8k/train_300.jsonl', help="Path to the dataset file.")
    parser.add_argument("--test_dataset_path", type=str, default='./datasets/gsm8k/test.jsonl', help="Path to the dataset file.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens for sampling.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of sampled responses per prompt.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    #save_hidden_states(args.model_name, args.data_name, args.train_dataset_path)

    # --- 加载训练集 ---
    heads_file =  '/home/zcy/hallucination_predict_by_prompt/feature/gsm8k_Qwen2.5-3B-Instruct_train_heads.npy'
    labels_file = '/home/zcy/hallucination_predict_by_prompt/feature/gsm8k_Qwen2.5-3B-Instruct_train_labels.npy'
    head_activations_train = np.load(heads_file)
    labels_train = np.load(labels_file)

    # --- 加载测试集 ---
    heads_file_test = f'./feature/{args.data_name}_{args.model_name}_test_baseline_heads.npy'
    labels_file_test = f'./feature/{args.data_name}_{args.model_name}_test_baseline_labels.npy'
    head_activations_test = np.load(heads_file_test)
    labels_test = np.load(labels_file_test)

    # --- 取第一个 token ---
    head_wise_activations_train = head_activations_train[:, :, 0, :, :]
    head_wise_activations_test = head_activations_test[:, :, 0, :, :]


    print(head_wise_activations_train.shape)
    print("训练集标签取值：", np.unique(labels_train))
    print("测试集标签取值：", np.unique(labels_test))
    print("head_activations_train shape:", head_wise_activations_train.shape)
    print("head_activations_test shape:", head_wise_activations_test.shape)
    layer_num = head_wise_activations_train.shape[1]
    head_num = head_wise_activations_train.shape[2]

    print('layer_num:', layer_num)
    print('head_num:', head_num)
    print("\n====== 训练集标签分布 ======")
    unique_train, counts_train = np.unique(labels_train, return_counts=True)
    for u, c in zip(unique_train, counts_train):
        print(f"标签 {u}: {c} ({c / len(labels_train):.2%})")

    print("\n====== 测试集标签分布 ======")
    unique_test, counts_test = np.unique(labels_test, return_counts=True)
    for u, c in zip(unique_test, counts_test):
        print(f"标签 {u}: {c} ({c / len(labels_test):.2%})")

    print("\n训练集样本数:", len(labels_train))
    print("测试集样本数:", len(labels_test))
    m_auc, m_auroc, d_clf = probe(layer_num, head_num, head_wise_activations_train, labels_train, head_wise_activations_test, labels_test, max_iter=500)

    print(m_auc)
    print("m_auc:", m_auc.max())

### baseline
# if __name__ == '__main__':
#     args = parse_args()
#     #save_hidden_states(args.model_name, args.data_name, args.train_dataset_path)
#     # 训练集
#     generate_and_save_features_vllm_baseline(
#         args.model_name,
#         args.data_name,
#         args.train_dataset_path,
#         split="train"
#     )

#     # # # 测试集
#     # generate_and_save_features_vllm_baseline(
#     #     args.model_name,
#     #     args.data_name,
#     #     args.test_dataset_path,
#     #     split="test"
#     # )
    
#     # --- 加载训练集特征 ---
#     heads_file = f'./feature/{args.data_name}_{args.model_name}_train_baseline_heads.npy'
#     layer_file = f'./feature/{args.data_name}_{args.model_name}_train_baseline_layers.npy'
#     labels_file = f'./feature/{args.data_name}_{args.model_name}_train_baseline_labels.npy'

#     head_activations_train = np.load(heads_file)
#     labels_train = np.load(labels_file)

#     # --- 加载测试集特征 ---
#     heads_file_test = f'./feature/{args.data_name}_{args.model_name}_test_baseline_heads.npy'
#     layer_file_test = f'./feature/{args.data_name}_{args.model_name}_test_baseline_layers.npy'
#     labels_file_test = f'./feature/{args.data_name}_{args.model_name}_test_baseline_labels.npy'

#     head_activations_test = np.load(heads_file_test)
#     labels_test = np.load(labels_file_test)

#     # --- 取第一 token 的表示 ---
#     head_wise_activations_train = head_activations_train[:, :, 0, :, :]
#     head_wise_activations_test = head_activations_test[:, :, 0, :, :]

#     # heads_file = '/home/zcy/hallucination_predict_by_prompt/feature/gsm8k_Qwen2.5-3B-Instruct_heads.npy'
#     # layer_file = '/home/zcy/hallucination_predict_by_prompt/feature/gsm8k_Qwen2.5-3B-Instruct_layers.npy'
#     # labels_file = '/home/zcy/hallucination_predict_by_prompt/feature/gsm8k_Qwen2.5-3B-Instruct_labels.npy'

#     # head_activations = np.load(heads_file)
#     # # layer_wise_activations = np.load(layer_file)
#     # labels = np.load(labels_file)

#     # head_activations = head_activations[:, :, 0, :, :]

#     # head_wise_activations_train = head_activations[:20, ...]
#     # labels_train = labels[:20]
#     # head_wise_activations_test = head_activations[10:, ...]
#     # labels_test = labels[10:]

    
#     # print("标签取值：", np.unique(labels))
#     # print("训练集标签取值：", np.unique(labels_train))
#     # print("测试集标签取值：", np.unique(labels_test))

#     # print(head_wise_activations_train.shape)
#     print("训练集标签取值：", np.unique(labels_train))
#     print("测试集标签取值：", np.unique(labels_test))
#     print("head_activations_train shape:", head_activations_train.shape)
#     print("head_activations_test shape:", head_activations_test.shape)
#     layer_num = head_wise_activations_train.shape[1]
#     head_num = head_wise_activations_train.shape[2]

#     print('layer_num:', layer_num)
#     print('head_num:', head_num)

#     m_auc, m_auroc, d_clf = probe(layer_num, head_num, head_wise_activations_train, labels_train, head_wise_activations_test, labels_test, max_iter=300)

#     print(m_auc)
#     print("m_auc:", m_auc.max())

### 添加测试集
# if __name__ == '__main__':
#     args = parse_args()

#     # --- 加载训练集 ---
#     heads_file = f'./feature/{args.data_name}_{args.model_name}_train_baseline_heads.npy'
#     labels_file = f'./feature/{args.data_name}_{args.model_name}_train_baseline_labels.npy'
#     head_activations_train = np.load(heads_file)
#     labels_train = np.load(labels_file)

#     # --- 加载测试集 ---
#     heads_file_test = f'./feature/{args.data_name}_{args.model_name}_test_baseline_heads.npy'
#     labels_file_test = f'./feature/{args.data_name}_{args.model_name}_test_baseline_labels.npy'
#     head_activations_test = np.load(heads_file_test)
#     labels_test = np.load(labels_file_test)

#     # --- 取第一个 token ---
#     head_wise_activations_train = head_activations_train[:, :, 0, :, :]
#     head_wise_activations_test = head_activations_test[:, :, 0, :, :]

#     print("训练集 shape:", head_wise_activations_train.shape)
#     print("测试集 shape:", head_wise_activations_test.shape)
#     print("训练集标签:", np.unique(labels_train))
#     print("测试集标签:", np.unique(labels_test))

#     layer_num = head_wise_activations_train.shape[1]
#     head_num = head_wise_activations_train.shape[2]

#     # --- 划分验证集 ---
#     X_val, X_test, y_val, y_test = train_test_split(
#         head_wise_activations_test,
#         labels_test,
#         test_size=0.8,       
#         random_state=42,
#         stratify=labels_test
#     )
#     print("训练集:", head_wise_activations_train.shape, 
#         "验证集:", X_val.shape, 
#         "测试集:", X_test.shape)


#     # --- 探测 ---
#     m_auc, m_auroc, m_val_auc, m_val_auroc, d_clf = probe_val(
#         layer_num, head_num,
#         head_wise_activations_train, labels_train,
#         X_val, y_val,
#         X_test, y_test,
#         max_iter=300
#     )

#     # --- 输出结果 ---
#     print("验证集最大AUC:", np.nanmax(m_val_auc))
#     print("测试集最大AUC:", np.nanmax(m_auc))

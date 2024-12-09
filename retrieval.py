import torch
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
import numpy as np
import pickle
from tqdm import tqdm
import os
import re
import json
import argparse
import logging
from collections import defaultdict

def encode_sequences_batch(sequences, tokenizer, model, device, batch_size=32):
    """Encode sequences in batches"""
    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

def parse_fasta_to_dict(fasta_file):
    """Parse FASTA file and return a dictionary"""
    logging.info(f"Parsing FASTA file: {fasta_file}")
    records = list(SeqIO.parse(fasta_file, "fasta"))
    result_dict = {}
    for record in records:
        description = record.description
        sequence = str(record.seq)
        parts = description.split(" | ")
        if len(parts) == 5:   # opt fasta has 5 fields
            protein_id, organism, ecnumber, pH, _ = parts
            result_dict[protein_id.strip()] = {
                'seq': sequence,
                'organism': organism.strip(),
                'ecnumber': ecnumber.strip(),
                'pH': pH.strip()
            }
        else:
            logging.warning(f"Unexpected format in FASTA file: {description}")
    logging.info(f"FASTA file parsed, {len(result_dict)} records found")
    return result_dict

def extract_and_save_features(fasta_file, output_file, tokenizer, model, device, batch_size=32):
    """Extract features and save them"""
    if os.path.exists(output_file):
        logging.info(f"Loading pre-extracted features from {output_file}")
        with open(output_file, 'rb') as f:
            features = pickle.load(f)
        records = parse_fasta_to_dict(fasta_file)
        return records, features

    records = parse_fasta_to_dict(fasta_file)
    sequences = [str(record['seq']) for record in records.values()]
    
    logging.info(f"Starting feature extraction for {len(sequences)} sequences")
    features = {}
    for i in tqdm(range(0, len(sequences), batch_size), desc=f"Extracting features {fasta_file}", unit="batch"):
        batch_sequences = sequences[i:i+batch_size]
        batch_embeddings = encode_sequences_batch(batch_sequences, tokenizer, model, device, batch_size)
        for j, embedding in enumerate(batch_embeddings):
            protein_id = list(records.keys())[i + j]
            features[protein_id] = embedding
    
    logging.info(f"Feature extraction complete, saving to {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(features, f)
    
    return records, features


def opt_retrieval(opt_features, opt_records, train_features=None, train_records=None, k=5, dataset_type="train"):
    """从opt训练集中检索序列
    
    Args:
        opt_features: 当前数据集的特征
        opt_records: 当前数据集的记录
        train_features: 训练集特征（用于验证/测试集）
        train_records: 训练集记录（用于验证/测试集）
        k: 检索数量
        dataset_type: 数据集类型（train/valid/test）
    """
    logging.info(f"Starting OPT retrieval for {dataset_type} dataset")
    opt_keys = list(opt_features.keys())
    
    if dataset_type == "train":
        # 训练集内部检索
        source_features = opt_features
        source_records = opt_records
        source_keys = opt_keys
    else:
        # 验证集和测试集从训练集中检索
        source_features = train_features
        source_records = train_records
        source_keys = list(train_features.keys())
    
    # 转换特征为numpy数组并归一化
    opt_matrix = np.array(list(opt_features.values()), dtype=np.float32)
    source_matrix = np.array(list(source_features.values()), dtype=np.float32)
    
    source_norm = source_matrix / np.linalg.norm(source_matrix, axis=1, keepdims=True)
    opt_norm = opt_matrix / np.linalg.norm(opt_matrix, axis=1, keepdims=True)

    # 计算余弦相似度矩阵
    cosine_similarity_matrix = np.dot(opt_norm, source_norm.T)
    
    results = []
    for i, opt_key in enumerate(tqdm(opt_keys, desc=f"Processing {dataset_type} sequences")):
        similarity_scores = cosine_similarity_matrix[i]
        
        if dataset_type == "train":
            # 训练集中排除自身
            similarity_scores[i] = -np.inf
        
        # 获取最相似的k个序列
        top_k_indices = similarity_scores.argsort()[-k:][::-1]
        top_k_source_keys = [source_keys[idx] for idx in top_k_indices]
        
        entry = {
            "opt_sequence": opt_records[opt_key]['seq'],
            "opt_id": opt_key,
            "opt_pH": opt_records[opt_key]['pH'],
            "env_sequences": [],  # 保持键名一致，实际存储的是OPT序列
            "env_ids": [],       # 保持键名一致，实际存储的是OPT ID
            "env_pHs": []        # 保持键名一致，实际存储的是OPT pH
        }
        
        for source_key in top_k_source_keys:
            entry["env_sequences"].append(source_records[source_key]['seq'])
            entry["env_ids"].append(source_key)
            entry["env_pHs"].append(source_records[source_key]['pH'])
            
        results.append(entry)
    
    logging.info(f"OPT retrieval completed for {dataset_type} dataset")
    return results

def opt_random_retrieval(opt_features, opt_records, train_features=None, train_records=None, k=5, dataset_type="train"):
    """从opt训练集中随机检索序列
    
    Args:
        opt_features: 当前数据集的特征
        opt_records: 当前数据集的记录
        train_features: 训练集特征（用于验证/测试集）
        train_records: 训练集记录（用于验证/测试集）
        k: 检索数量
        dataset_type: 数据集类型（train/valid/test）
    """
    logging.info(f"Starting OPT random retrieval for {dataset_type} dataset")
    opt_keys = list(opt_features.keys())
    
    if dataset_type == "train":
        source_records = opt_records
        source_keys = opt_keys
    else:
        source_records = train_records
        source_keys = list(train_features.keys())
    
    results = []
    for i, opt_key in enumerate(tqdm(opt_keys, desc=f"Processing {dataset_type} sequences")):
        available_keys = source_keys.copy()
        if dataset_type == "train":
            # 训练集中排除自身
            available_keys.remove(opt_key)
            
        # 随机选择k个序列
        selected_keys = np.random.choice(available_keys, size=k, replace=False)
        
        entry = {
            "opt_sequence": opt_records[opt_key]['seq'],
            "opt_id": opt_key,
            "opt_pH": opt_records[opt_key]['pH'],
            "env_sequences": [],
            "env_ids": [],
            "env_pHs": []
        }
        
        for source_key in selected_keys:
            entry["env_sequences"].append(source_records[source_key]['seq'])
            entry["env_ids"].append(source_key)
            entry["env_pHs"].append(source_records[source_key]['pH'])
            
        results.append(entry)
    
    logging.info(f"OPT random retrieval completed for {dataset_type} dataset")
    return results

def opt_fixed_random_retrieval(opt_features, opt_records, train_features=None, train_records=None, k=5, dataset_type="train"):
    """为所有opt选择相同的k个随机训练集序列
    
    Args:
        opt_features: 当前数据集的特征
        opt_records: 当前数据集的记录
        train_features: 训练集特征（用于验证/测试集）
        train_records: 训练集记录（用于验证/测试集）
        k: 检索数量
        dataset_type: 数据集类型（train/valid/test）
    """
    logging.info(f"Starting OPT fixed random retrieval for {dataset_type} dataset")
    opt_keys = list(opt_features.keys())
    
    if dataset_type == "train":
        source_records = opt_records
        source_keys = opt_keys
    else:
        source_records = train_records
        source_keys = list(train_features.keys())
    
    # 固定随机种子以确保可重复性
    np.random.seed(42)
    
    # 检查固定source keys的缓存文件是否存在
    cache_file = "data/fixed_random_opt_keys.json"
    if dataset_type == "train" or not os.path.exists(cache_file):
        # 只在训练集处理时或缓存不存在时生成新的随机选择
        fixed_source_keys = np.random.choice(source_keys, size=k, replace=False).tolist()
        # 保存选择的keys
        with open(cache_file, 'w') as f:
            json.dump(fixed_source_keys, f)
        logging.info(f"Generated and saved new fixed opt keys: {fixed_source_keys}")
    else:
        # 读取之前保存的固定选择
        with open(cache_file, 'r') as f:
            fixed_source_keys = json.load(f)
        logging.info(f"Loaded existing fixed opt keys: {fixed_source_keys}")
    
    results = []
    for opt_key in tqdm(opt_keys, desc=f"Processing {dataset_type} sequences"):
        entry = {
            "opt_sequence": opt_records[opt_key]['seq'],
            "opt_id": opt_key,
            "opt_pH": opt_records[opt_key]['pH'],
            "env_sequences": [],
            "env_ids": [],
            "env_pHs": []
        }
        
        for source_key in fixed_source_keys:
            if source_key != opt_key:  # 避免在训练集中选到自身
                entry["env_sequences"].append(source_records[source_key]['seq'])
                entry["env_ids"].append(source_key)
                entry["env_pHs"].append(source_records[source_key]['pH'])
            
        results.append(entry)
    
    logging.info(f"OPT fixed random retrieval completed for {dataset_type} dataset")
    return results

def opt_retrieval_scaled(opt_features, opt_records, train_features=None, train_records=None, k=5, dataset_type="train", scale_ratio=1.0):
    """从缩小规模后的opt训练集中检索序列,只缩小训练集规模"""
    logging.info(f"Starting scaled OPT retrieval for {dataset_type} dataset with {scale_ratio*100}% training data")
    opt_keys = list(opt_features.keys())
    
    if dataset_type == "train":
        # 只对训练集进行缩小
        np.random.seed(42)
        num_samples = int(len(opt_keys) * scale_ratio)
        selected_opt_keys = np.random.choice(opt_keys, size=num_samples, replace=False).tolist()
        
        selected_opt_features = {k: opt_features[k] for k in selected_opt_keys}
        selected_opt_records = {k: opt_records[k] for k in selected_opt_keys}
        
        source_features = selected_opt_features
        source_records = selected_opt_records
        source_keys = selected_opt_keys
        
        # 保存缩小后的训练集keys供验证集和测试集使用
        cache_file = f"data/scaled_cache/scaled_train_keys_{scale_ratio}.json"
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(source_keys, f)
    else:
        # 验证集和测试集保持原始大小，但从缩小后的训练集中检索
        selected_opt_features = opt_features
        selected_opt_records = opt_records
        selected_opt_keys = opt_keys
        
        cache_file = f"data/scaled_cache/scaled_train_keys_{scale_ratio}.json"
        with open(cache_file, 'r') as f:
            source_keys = json.load(f)
            source_features = {k: train_features[k] for k in source_keys}
            source_records = {k: train_records[k] for k in source_keys}
    
    # 计算余弦相似度
    opt_matrix = np.array(list(selected_opt_features.values()), dtype=np.float32)
    source_matrix = np.array(list(source_features.values()), dtype=np.float32)
    
    source_norm = source_matrix / np.linalg.norm(source_matrix, axis=1, keepdims=True)
    opt_norm = opt_matrix / np.linalg.norm(opt_matrix, axis=1, keepdims=True)
    cosine_similarity_matrix = np.dot(opt_norm, source_norm.T)
    
    results = []
    for i, opt_key in enumerate(tqdm(selected_opt_keys, desc=f"Processing {dataset_type} sequences")):
        similarity_scores = cosine_similarity_matrix[i]
        
        if dataset_type == "train" and opt_key in source_keys:
            similarity_scores[list(source_keys).index(opt_key)] = -np.inf
        
        top_k_indices = similarity_scores.argsort()[-k:][::-1]
        top_k_source_keys = [source_keys[idx] for idx in top_k_indices]
        
        entry = {
            "opt_sequence": selected_opt_records[opt_key]['seq'],
            "opt_id": opt_key,
            "opt_pH": selected_opt_records[opt_key]['pH'],
            "env_sequences": [source_records[k]['seq'] for k in top_k_source_keys],
            "env_ids": top_k_source_keys,
            "env_pHs": [source_records[k]['pH'] for k in top_k_source_keys]
        }
        results.append(entry)
    
    logging.info(f"Scaled OPT retrieval completed for {dataset_type} dataset with {len(results)} sequences")
    return results

def process_dataset(opt_file,  output_json, tokenizer, model, device, topk=5, 
                   batch_size=100, opt_features_file=None, 
                   strategy=None, dataset_type=None,
                   train_features=None, train_records=None):
    """Process dataset"""
    logging.info(f"Starting to process dataset: {opt_file}")
    opt_records, opt_features = extract_and_save_features(
        opt_file, 
        opt_features_file or f"{os.path.splitext(opt_file)[0]}_features.pkl", 
        tokenizer, 
        model, 
        device, 
        batch_size
    )

    logging.info(f"Starting to find similar sequences using strategy: {strategy}...")
    if strategy == "opt_retrieval":
        if train_features is None and dataset_type != "train":
            raise ValueError("train_features is required for opt_retrieval strategy on validation/test sets")
        results = opt_retrieval(opt_features, opt_records, train_features, train_records, topk, dataset_type)
    elif strategy == "opt_random":
        results = opt_random_retrieval(opt_features, opt_records, train_features, train_records, topk, dataset_type)
    elif strategy == "opt_fixed_random":
        results = opt_fixed_random_retrieval(opt_features, opt_records, train_features, train_records, topk, dataset_type)
    elif strategy.startswith("opt_retrieval_scaled"):
    # 从策略名称中提取比例，例如 "opt_retrieval_scaled_0.2" 表示使用20%的数据
        scale_ratio = float(strategy.split("_")[-1])
        results = opt_retrieval_scaled(opt_features, opt_records, train_features, train_records, topk, dataset_type, scale_ratio)
    else:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")

    logging.info(f"Saving processed data to {output_json}")
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"Processed data saved to {output_json}")

def main():
    parser = argparse.ArgumentParser(description="Process protein sequences and find similar environmental sequences.")
    parser.add_argument("--opt_train", default="data/phopt_training.fasta", help="Path to the OPT training FASTA file")
    parser.add_argument("--opt_test", default="data/phopt_testing.fasta", help="Path to the OPT testing FASTA file")
    parser.add_argument("--opt_valid", default="data/phopt_validation.fasta", help="Path to the OPT validation FASTA file")
    parser.add_argument("--output_dir", default="data/processed", help="Base directory to save output JSON files")
    parser.add_argument("--topk", type=int, default=5, help="Number of similar sequences to retrieve")
    parser.add_argument("--batch_size", type=int, default=100, help="Size of batches for processing")
    parser.add_argument("--features_dir", default="data/features", help="Directory to save feature files")
    parser.add_argument("--model_name", default="path_to/facebook/esm2_t33_650M_UR50D", help="Path to the ESM2 model")
    parser.add_argument("--strategy", default="opt_retrieval", 
                       choices=["opt_retrieval", "opt_random", "opt_fixed_random",
                               "opt_retrieval_scaled_0.2","opt_retrieval_scaled_0.6"],  
                                help="Retrieval strategy to use")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load ESM2 model and tokenizer
    logging.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    logging.info("Model loaded successfully")

    logging.info("Starting main program execution")
    output_dir = os.path.join(args.output_dir, f"top{args.topk}", f"esm2_{args.strategy}")
    logging.info(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Creating features directory: {args.features_dir}")
    os.makedirs(args.features_dir, exist_ok=True)

    
    # 预加载训练集特征（用于opt策略）
    train_features = None
    train_records = None

    train_features_file = os.path.join(args.features_dir, "opt_train_features.pkl")
    train_records, train_features = extract_and_save_features(
        args.opt_train, 
        train_features_file, 
        tokenizer, 
        model, 
        device, 
        args.batch_size
    )

    # 处理每个数据集
    for dataset, opt_file in [("train", args.opt_train), ("test", args.opt_test), ("valid", args.opt_valid)]:
        output_json = os.path.join(output_dir, f"retrieval_{dataset}.json")
        opt_features_file = os.path.join(args.features_dir, f"opt_{dataset}_features.pkl")
        
        logging.info(f"Starting to process {dataset} dataset...")
        process_dataset(
            opt_file, 
            output_json, 
            tokenizer, 
            model, 
            device,
            args.topk, 
            args.batch_size, 
            opt_features_file, 
            args.strategy, 
            args.lambda_param,
            dataset_type=dataset,
            train_features=train_features,
            train_records=train_records
        )
    
    logging.info("All datasets processed and saved.")

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    





# def mmr_ph_diverse_retrieval(opt_features, env_features, opt_records, env_records, k=5, lambda_param=0.5):
#     """Retrieval method based on MMR and pH diversity"""
#     logging.info("Starting MMR-pH diversity retrieval")
#     opt_keys = list(opt_features.keys())
#     env_keys = list(env_features.keys())
    
#     # Convert features to numpy arrays and normalize
#     opt_matrix = np.array(list(opt_features.values()), dtype=np.float32)
#     env_matrix = np.array(list(env_features.values()), dtype=np.float32)
#     env_norm = env_matrix / np.linalg.norm(env_matrix, axis=1, keepdims=True)
#     opt_norm = opt_matrix / np.linalg.norm(opt_matrix, axis=1, keepdims=True)

#     # Calculate cosine similarity matrix
#     cosine_similarity_matrix = np.dot(opt_norm, env_norm.T)
    
#     results = []
#     for i, opt_key in enumerate(tqdm(opt_keys, desc="Processing opt_keys")):
#         similarity_scores = cosine_similarity_matrix[i]
        
#         selected_env_keys = []
#         remaining_env_keys = env_keys.copy()
        
#         while len(selected_env_keys) < k and remaining_env_keys:
#             mmr_scores = {}
#             for env_key in remaining_env_keys:
#                 env_index = env_keys.index(env_key)
#                 # Calculate relevance (cosine similarity)
#                 relevance = similarity_scores[env_index]
                
#                 if not selected_env_keys:
#                     diversity = 1  # Maximum diversity for the first selection
#                 else:
#                     # Calculate pH diversity
#                     selected_pHs = [float(env_records[key]['pH']) for key in selected_env_keys]
#                     current_pH = float(env_records[env_key]['pH'])
#                     pH_differences = [abs(current_pH - pH) for pH in selected_pHs]
#                     diversity = min(pH_differences) / 14  # Normalize pH difference
                
#                 # Calculate MMR score
#                 mmr_scores[env_key] = lambda_param * relevance + (1 - lambda_param) * diversity
            
#             # Select the env_key with the highest MMR score
#             best_env_key = max(mmr_scores, key=mmr_scores.get)
#             selected_env_keys.append(best_env_key)
#             remaining_env_keys.remove(best_env_key)
#             print(selected_env_keys)
        
#         # Prepare the result entry
#         entry = {
#             "opt_sequence": opt_records[opt_key]['seq'],
#             "opt_id": opt_key,
#             "opt_pH": opt_records[opt_key]['pH'],
#             "env_sequences": [],
#             "env_ids": [],
#             "env_pHs": []
#         }
#         for env_key in selected_env_keys:
#             entry["env_sequences"].append(env_records[env_key]['seq'])
#             entry["env_ids"].append(env_key)
#             entry["env_pHs"].append(env_records[env_key]['pH'])
#         results.append(entry)
    
#     logging.info("MMR-pH diversity retrieval completed")
#     return results
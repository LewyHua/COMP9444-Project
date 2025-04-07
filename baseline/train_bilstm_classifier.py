#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于"Movie Genre Classification from Plot Summaries Using Bidirectional LSTM"论文
实现电影类型分类模型，使用预处理好的MPST_v2数据集训练BiLSTM模型
"""

import os
import json
import numpy as np
import pandas as pd
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, hamming_loss
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter
from tqdm import tqdm
import re
import seaborn as sns
from baseline.config import (
    DATA_DIR, OUTPUT_DIR,
    VALID_TAGS_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH,
    BILSTM_MODEL_PATH, BILSTM_VOCAB_PATH, BILSTM_RESULTS_PATH,
    TEXT_LENGTH_DIST_PATH, BILSTM_TOP_TAGS_PERFORMANCE_PATH, BILSTM_LOSS_CURVE_PATH,
    SEED, MAX_VOCAB_SIZE, MAX_SEQ_LENGTH, BATCH_SIZE, 
    EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, 
    LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE,
    PREDICTION_THRESHOLD  # 添加预测阈值
)

# 设置matplotlib不使用中文字体，防止警告
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.unicode_minus'] = False
# 强制使用特定的字体文件
plt.rcParams['svg.fonttype'] = 'none'

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 创建新的输出目录
BILSTM_3TAGS_DIR = 'D:/share/base/bilstm-3tags'
os.makedirs(BILSTM_3TAGS_DIR, exist_ok=True)

# 新的输出路径
BILSTM_BEST_MODEL_PATH = os.path.join(BILSTM_3TAGS_DIR, 'best_model.pt')
BILSTM_WORST_MODEL_PATH = os.path.join(BILSTM_3TAGS_DIR, 'worst_model.pt')
BILSTM_3TAGS_RESULTS_PATH = os.path.join(BILSTM_3TAGS_DIR, 'results.json')
BILSTM_3TAGS_LOSS_CURVE_PATH = os.path.join(BILSTM_3TAGS_DIR, 'loss_curves.png')
BILSTM_3TAGS_CONFUSION_MATRIX_PATH = os.path.join(BILSTM_3TAGS_DIR, 'confusion_matrix.png')
BILSTM_3TAGS_TAG_CORR_PATH = os.path.join(BILSTM_3TAGS_DIR, 'tag_correlations.png')
BILSTM_3TAGS_PERFORMANCE_DIST_PATH = os.path.join(BILSTM_3TAGS_DIR, 'performance_distribution.png')
BILSTM_3TAGS_TOP_TAGS_PATH = os.path.join(BILSTM_3TAGS_DIR, 'top_tags_performance.png')
BILSTM_3TAGS_EPOCH_METRICS_PATH = os.path.join(BILSTM_3TAGS_DIR, 'epoch_metrics.png')
BILSTM_3TAGS_TAG_HEATMAP_PATH = os.path.join(BILSTM_3TAGS_DIR, 'tag_heatmap.png')

# 设置随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class MovieDataset(Dataset):
    """电影简介数据集"""
    
    def __init__(self, texts, labels, word_to_idx, max_length=MAX_SEQ_LENGTH):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # 将文本转换为索引序列
        if not isinstance(text, str):
            text = ""
        tokens = text.split()
        
        # 截断或填充到固定长度
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in tokens]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

def collate_fn(batch):
    """处理不同长度的序列"""
    texts, labels = zip(*batch)
    # 对序列进行填充
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    return texts_padded, torch.stack(labels)

class AttentionLayer(nn.Module):
    """注意力层，基于论文中的描述"""
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: [batch_size, seq_len, hidden_dim]
        
        # 计算注意力权重
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attention_weights shape: [batch_size, seq_len, 1]
        
        # 应用注意力权重
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        # context_vector shape: [batch_size, hidden_dim]
        
        return context_vector, attention_weights

class BiLSTMWithAttention(nn.Module):
    """带有注意力机制的双向LSTM模型，基于论文实现"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.5):
        super(BiLSTMWithAttention, self).__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 双向LSTM层
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            bidirectional=True, 
                            dropout=dropout if n_layers > 1 else 0,
                            batch_first=True)
        
        # 注意力层
        self.attention = AttentionLayer(hidden_dim * 2)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数
        self.relu = nn.ReLU()
        
    def forward(self, text):
        # text shape: [batch size, seq len]
        
        # 嵌入层
        embedded = self.embedding(text)
        # embedded shape: [batch size, seq len, embedding dim]
        
        # LSTM层
        lstm_output, (hidden, cell) = self.lstm(embedded)
        # lstm_output shape: [batch size, seq len, hidden dim * 2]
        
        # 注意力机制
        context_vector, attention_weights = self.attention(lstm_output)
        # context_vector shape: [batch size, hidden dim * 2]
        
        # 全连接层
        dense1 = self.relu(self.fc1(self.dropout(context_vector)))
        # dense1 shape: [batch size, hidden dim]
        
        output = self.fc2(self.dropout(dense1))
        # output shape: [batch size, output dim]
        
        return output

def load_data():
    """加载预处理好的数据"""
    print("正在加载数据...")
    
    # 加载标签
    with open(VALID_TAGS_PATH, 'r') as f:
        valid_tags = json.load(f)
    
    # 创建标签列表
    tag_columns = [f"tag_{tag.replace(' ', '_')}" for tag in valid_tags]
    
    # 加载数据集
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    # 提取文本和标签
    X_train = train_df['processed_text'].values
    X_val = val_df['processed_text'].values
    X_test = test_df['processed_text'].values
    
    y_train = train_df[tag_columns].values
    y_val = val_df[tag_columns].values
    y_test = test_df[tag_columns].values
    
    print(f"数据加载完成:")
    print(f"训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_val)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    print(f"标签数量: {len(valid_tags)}")
    
    # 统计文本长度
    text_lengths = [len(text.split()) for text in X_train if isinstance(text, str)]
    plt.figure(figsize=(10, 6))
    plt.hist(text_lengths, bins=50)
    plt.title('Text Length Distribution')
    plt.xlabel('Length (words)')
    plt.ylabel('Count')
    plt.savefig(TEXT_LENGTH_DIST_PATH)
    
    avg_length = np.mean(text_lengths)
    median_length = np.median(text_lengths)
    print(f"平均文本长度: {avg_length:.2f} 词")
    print(f"中位文本长度: {median_length:.2f} 词")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'valid_tags': valid_tags,
        'tag_columns': tag_columns
    }

def build_vocabulary(texts, max_vocab_size=MAX_VOCAB_SIZE):
    """构建词汇表，基于论文中的词汇处理方法"""
    print("正在构建词汇表...")
    
    # 特殊标记
    special_tokens = ['<PAD>', '<UNK>']
    
    # 统计词频
    word_counts = Counter()
    for text in tqdm(texts, desc="统计词频"):
        if isinstance(text, str):
            words = text.split()
            word_counts.update(words)
    
    # 选择最常见的词
    most_common = word_counts.most_common(max_vocab_size - len(special_tokens))
    vocab = special_tokens + [word for word, _ in most_common]
    
    # 创建词到索引的映射
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    
    print(f"词汇表构建完成，大小: {len(vocab)}")
    
    # 保存词汇表
    with open(BILSTM_VOCAB_PATH, 'w') as f:
        json.dump(word_to_idx, f)
    
    return vocab, word_to_idx

def train(model, train_loader, val_loader, criterion, optimizer, n_epochs=MAX_EPOCHS, patience=PATIENCE):
    """训练模型"""
    print("\n开始训练模型...")
    
    # 用于早停和保存最佳/最差模型
    best_val_f1 = 0
    worst_val_f1 = float('inf')
    no_improvement = 0
    best_model = None
    worst_model = None
    
    # 保存损失和F1分数的历史
    train_losses = []
    val_losses = []
    val_f1s = []
    train_f1s = []
    epoch_nums = []
    
    # 训练循环
    for epoch in range(1, n_epochs + 1):
        epoch_nums.append(epoch)
        # 训练模式
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")
        
        all_train_labels = []
        all_train_preds = []
        
        for batch in progress_bar:
            # 获取批次数据
            texts, labels = batch
            texts = texts.to(device)
            labels = labels.to(device)
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            predictions = model(texts)
            
            # 计算损失
            loss = criterion(predictions, labels)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 累积批次损失
            epoch_loss += loss.item()
            
            # 收集训练指标
            train_preds = (torch.sigmoid(predictions) > PREDICTION_THRESHOLD).float()
            all_train_labels.append(labels.cpu().numpy())
            all_train_preds.append(train_preds.cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # 计算训练集的F1分数
        train_labels = np.vstack(all_train_labels)
        train_preds = np.vstack(all_train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='samples', zero_division=0)
        train_f1s.append(train_f1)
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 评估验证集
        model.eval()
        val_loss = 0
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                texts, labels = batch
                texts = texts.to(device)
                labels = labels.to(device)
                
                predictions = model(texts)
                loss = criterion(predictions, labels)
                val_loss += loss.item()
                
                # 获取概率和二值预测
                probs = torch.sigmoid(predictions)
                preds = (probs > PREDICTION_THRESHOLD).float()
                
                # 将预测和标签添加到列表
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 计算F1分数
        val_labels = np.vstack(all_labels)
        val_preds = np.vstack(all_preds)
        val_probs = np.vstack(all_probs)
        
        # 样本级别F1分数
        sample_f1 = f1_score(val_labels, val_preds, average='samples', zero_division=0)
        val_f1s.append(sample_f1)
        
        # 打印进度
        print(f'Epoch {epoch}/{n_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val F1: {sample_f1:.4f}')
        
        # 检查是否是最佳模型
        if sample_f1 > best_val_f1:
            best_val_f1 = sample_f1
            no_improvement = 0
            best_model = model.state_dict().copy()
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'train_f1': train_f1,
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss
            }, BILSTM_BEST_MODEL_PATH)
            print(f'  新的最佳F1分数！已保存到 {BILSTM_BEST_MODEL_PATH}')
        else:
            no_improvement += 1
            print(f'  没有改进: {no_improvement}/{patience}')
        
        # 检查是否是最差模型（第一个epoch后）
        if epoch > 1 and sample_f1 < worst_val_f1:
            worst_val_f1 = sample_f1
            worst_model = model.state_dict().copy()
            # 保存最差模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': worst_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': worst_val_f1,
                'train_f1': train_f1,
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss
            }, BILSTM_WORST_MODEL_PATH)
            print(f'  新的最差F1分数！已保存到 {BILSTM_WORST_MODEL_PATH}')
            
    # 绘制训练曲线 - 增强版
    plt.figure(figsize=(15, 10))
    
    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(epoch_nums, train_losses, 'b-', label='Training Loss')
    plt.plot(epoch_nums, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    
    # F1分数曲线
    plt.subplot(2, 2, 2)
    plt.plot(epoch_nums, train_f1s, 'g-', label='Training F1')
    plt.plot(epoch_nums, val_f1s, 'm-', label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Curves')
    plt.legend()
    plt.grid(True)
    
    # 训练vs验证损失
    plt.subplot(2, 2, 3)
    plt.scatter(train_losses, val_losses, alpha=0.7)
    plt.xlabel('Training Loss')
    plt.ylabel('Validation Loss')
    plt.title('Training vs Validation Loss')
    for i, epoch in enumerate(epoch_nums):
        plt.annotate(str(epoch), (train_losses[i], val_losses[i]))
    plt.grid(True)
    
    # 训练vs验证F1
    plt.subplot(2, 2, 4)
    plt.scatter(train_f1s, val_f1s, alpha=0.7)
    plt.xlabel('Training F1')
    plt.ylabel('Validation F1')
    plt.title('Training vs Validation F1')
    for i, epoch in enumerate(epoch_nums):
        plt.annotate(str(epoch), (train_f1s[i], val_f1s[i]))
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(BILSTM_3TAGS_LOSS_CURVE_PATH)
    plt.close()
    
    # 保存训练指标数据
    training_metrics = {
        'epochs': epoch_nums,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_f1': train_f1s,
        'val_f1': val_f1s,
        'best_epoch': np.argmax(val_f1s) + 1,
        'best_val_f1': best_val_f1,
        'worst_epoch': np.argmin(val_f1s[1:]) + 2,  # +2 because we skip the first epoch
        'worst_val_f1': worst_val_f1
    }
    
    with open(os.path.join(BILSTM_3TAGS_DIR, 'training_metrics.json'), 'w') as f:
        # 转换numpy值为Python原生类型
        metrics_json = {k: v if not isinstance(v, (np.ndarray, np.float32, np.float64, np.int64)) 
                        else v.tolist() if hasattr(v, 'tolist') else float(v) 
                        for k, v in training_metrics.items()}
        json.dump(metrics_json, f, indent=2)
    
    # 加载最佳模型
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model, best_val_f1, training_metrics

def evaluate_multilabel(labels, preds):
    """Calculate three key evaluation metrics for multilabel classification"""
    # Calculate Micro-F1 score (treats all label instances as a whole)
    micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
    
    # Calculate Tag Recall (proportion of tags correctly predicted by the model)
    # Compute whether each tag can be recognized by the model (predicted at least once)
    tag_recall = 0
    tags_learned = 0
    
    # Count occurrences of each tag in true labels
    true_tag_presence = np.sum(labels, axis=0) > 0
    # Count occurrences of each tag in predictions
    pred_tag_presence = np.sum(preds, axis=0) > 0
    
    # Check if each tag is correctly predicted at least once
    correct_tag_predictions = np.zeros(labels.shape[1], dtype=bool)
    for i in range(labels.shape[1]):
        # For each tag, check if there are samples where both true and predicted labels are 1
        tag_true_samples = labels[:, i] == 1
        if np.any(tag_true_samples):
            tag_correct_preds = np.logical_and(labels[:, i] == 1, preds[:, i] == 1)
            correct_tag_predictions[i] = np.any(tag_correct_preds)
    
    # Total number of tags used in true labels
    total_tags_used = np.sum(true_tag_presence)
    # Number of tags correctly learned
    tags_learned = np.sum(correct_tag_predictions)
    # Tag recall rate
    tag_recall = float(tags_learned) / float(total_tags_used) if total_tags_used > 0 else 0.0
    
    # Return a dictionary of the three metrics
    return {
        'micro_f1_score': float(micro_f1),
        'tag_recall': float(tag_recall),
        'tags_learned': int(tags_learned),
        'total_tags_used': int(total_tags_used)
    }

def evaluate(model, data_loader, criterion, tag_names, device):
    """Evaluate model performance"""
    epoch_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Define Top-k and minimum probability threshold
    MIN_PROB_THRESHOLD = 0.1  # Minimum probability threshold, adjustable
    k = 3  # Select top-k labels
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for batch in data_loader:
            # Get data - make sure batch is handled as a tuple
            texts, labels = batch
            texts = texts.to(device)
            labels = labels.to(device)
            
            # Get predictions
            predictions = model(texts)
            
            # Calculate loss
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(predictions)
            
            # Method 1: Using threshold
            preds_threshold = (probs > PREDICTION_THRESHOLD).float()
            
            # Method 2: Top-k + minimum probability threshold
            # Get indices and probability values for Top-k
            top_probs, top_indices = torch.topk(probs, k=k, dim=1)
            
            # Initialize prediction matrix
            preds_topk = torch.zeros_like(probs)
            
            # Fill in predictions for each sample that meet the criteria
            for i, (indices, values) in enumerate(zip(top_indices, top_probs)):
                # Only keep labels with probability above threshold
                valid_mask = values >= MIN_PROB_THRESHOLD
                valid_indices = indices[valid_mask]
                
                # Set qualifying labels to 1
                if len(valid_indices) > 0:
                    preds_topk[i, valid_indices] = 1.0
            
            # Use Top-k + minimum probability threshold method for predictions
            preds = preds_topk
            
            # Collect predictions and labels
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # Combine results from all batches
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    
    # Calculate traditional metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='samples', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='samples', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='samples', zero_division=0)
    
    # Calculate per-label metrics
    per_label_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_label_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_label_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    per_label_support = np.sum(all_labels, axis=0)
    
    # Calculate multilabel evaluation metrics
    multilabel_metrics = evaluate_multilabel(all_labels, all_preds)
    micro_f1 = multilabel_metrics['micro_f1_score']
    tag_recall = multilabel_metrics['tag_recall']
    tags_learned = multilabel_metrics['tags_learned']
    total_tags_used = multilabel_metrics['total_tags_used']
    
    # Print overall evaluation metrics
    print(f"Evaluation Results (Top-{k} + Min Threshold={MIN_PROB_THRESHOLD}):")
    print(f"Exact Match Accuracy: {accuracy:.4f}")
    print(f"Sample Precision: {precision:.4f}")
    print(f"Sample Recall: {recall:.4f}")
    print(f"Sample F1 Score: {f1:.4f}")
    print(f"Micro-F1 Score: {micro_f1:.4f}")
    print(f"Tag Recall: {tag_recall:.4f} ({tags_learned}/{total_tags_used} tags learned)")
    
    # Organize results as a dictionary
    performance = {
        'loss': epoch_loss / len(data_loader),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'micro_f1': float(micro_f1),
        'tag_recall': float(tag_recall),
        'tags_learned': int(tags_learned),
        'total_tags_used': int(total_tags_used),
        'multilabel_metrics': multilabel_metrics,  # Add combined metrics
        'per_label': {},
        'prediction_method': 'top_k_with_threshold',
        'k_value': k,
        'min_threshold': MIN_PROB_THRESHOLD
    }
    
    # Add performance metrics for each label
    for i, tag in enumerate(tag_names):
        performance['per_label'][tag] = {
            'precision': float(per_label_precision[i]),
            'recall': float(per_label_recall[i]),
            'f1': float(per_label_f1[i]),
            'support': int(per_label_support[i])
        }
    
    return performance, (all_labels, all_preds)

def plot_top_tags_performance(evaluation_results, top_n=20, save_path=None):
    """Plot bar chart for tag performance"""
    # Extract label F1 scores
    per_label = evaluation_results['per_label']
    
    # Prepare data
    tags = []
    f1_scores = []
    supports = []
    
    # Sort by F1 score
    sorted_tags = sorted(
        per_label.items(), 
        key=lambda x: x[1]['f1'], 
        reverse=True
    )
    
    # Select top N tags
    for tag, metrics in sorted_tags[:top_n]:
        tags.append(tag)
        f1_scores.append(metrics['f1'])
        supports.append(metrics['support'])
    
    # Create chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set x-axis positions
    x = np.arange(len(tags))
    
    # Draw bar chart
    bars = ax.bar(x, f1_scores, width=0.6)
    
    # Add labels to each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'Support: {supports[i]}', ha='center', va='bottom', rotation=0)
    
    # Set title and labels
    ax.set_title(f'Top {top_n} Tags Performance (F1 Score)', fontsize=15)
    ax.set_xlabel('Tags', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save chart if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Top tags performance chart saved to: {save_path}")
    else:
        plt.savefig(BILSTM_3TAGS_TOP_TAGS_PATH, dpi=300, bbox_inches='tight')
        print(f"Top tags performance chart saved to: {BILSTM_3TAGS_TOP_TAGS_PATH}")
    
    plt.close()

def save_model_and_results(model, word_to_idx, val_evaluation, test_evaluation):
    """保存模型、词汇表和评估结果"""
    print("\n正在保存模型和结果...")
    
    # 保存模型
    torch.save(model.state_dict(), BILSTM_MODEL_PATH)
    
    # 保存结果
    results = {
        'model_type': 'bilstm',
        'validation': val_evaluation,
        'test': test_evaluation,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 保存为JSON
    results_path = BILSTM_RESULTS_PATH
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"模型和结果已保存到: {OUTPUT_DIR}")

def plot_new_metrics(val_metrics, test_metrics, save_path):
    """Plot visualization for the new evaluation metrics"""
    # Prepare data
    metrics = ['micro_f1_score', 'tag_recall']
    metric_names = ['Micro-F1 Score', 'Tag Recall']
    
    val_values = [val_metrics[m] for m in metrics]
    test_values = [test_metrics[m] for m in metrics]
    
    # Create chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # First subplot: Comparison of Micro-F1 and Tag Recall
    x = np.arange(len(metrics))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, val_values, width, label='Validation Set', color='steelblue')
    rects2 = ax1.bar(x + width/2, test_values, width, label='Test Set', color='lightcoral')
    
    # Set labels and title for the first subplot
    ax1.set_title('Evaluation Metrics Comparison', fontsize=14)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names, fontsize=12)
    ax1.legend()
    
    # Add labels to the bar chart
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    # Second subplot: Tags Learned vs Total Tags
    tags_data = [
        val_metrics['tags_learned'], 
        val_metrics['total_tags_used'] - val_metrics['tags_learned'],
        test_metrics['tags_learned'],
        test_metrics['total_tags_used'] - test_metrics['tags_learned']
    ]
    
    labels = ['Learned', 'Not Learned']
    x_labels = ['Validation Set', 'Test Set']
    
    x2 = np.arange(len(x_labels))
    width2 = 0.35
    
    ax2.bar(x2, [val_metrics['tags_learned'], test_metrics['tags_learned']], 
            width2, label='Tags Learned', color='forestgreen')
    ax2.bar(x2, [val_metrics['total_tags_used'] - val_metrics['tags_learned'], 
                test_metrics['total_tags_used'] - test_metrics['tags_learned']], 
            width2, bottom=[val_metrics['tags_learned'], test_metrics['tags_learned']], 
            label='Tags Not Learned', color='lightgray')
    
    # Set labels and title for the second subplot
    ax2.set_title('Tag Learning Status', fontsize=14)
    ax2.set_ylabel('Number of Tags', fontsize=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(x_labels, fontsize=12)
    ax2.legend()
    
    # Add percentage labels to the stacked bar chart
    for i, x_pos in enumerate(x2):
        total = val_metrics['total_tags_used'] if i == 0 else test_metrics['total_tags_used']
        learned = val_metrics['tags_learned'] if i == 0 else test_metrics['tags_learned']
        percentage = learned / total * 100 if total > 0 else 0
        
        ax2.text(x_pos, total/2, f'{percentage:.1f}%', 
                ha='center', va='center', fontweight='bold')
        
        # Add specific numbers
        ax2.text(x_pos, total + 1, f'{learned}/{total}', 
                ha='center', va='bottom')
    
    # Beautify the chart
    for ax in [ax1, ax2]:
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Adjust layout and save
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Evaluation metrics comparison chart saved to: {save_path}")

def main():
    """Main function"""
    print("\n========== Starting BiLSTM Movie Genre Classification Model Training (Based on Paper Implementation) ==========")
    start_time = time.time()
    print(f"Training results will be saved to: {BILSTM_3TAGS_DIR}")
    print(f"Set to train for {MAX_EPOCHS} epochs")
    
    # Load data
    data = load_data()
    valid_tags = data['valid_tags']
    
    # Build vocabulary
    vocab, word_to_idx = build_vocabulary(data['X_train'])
    
    # Statistics on label distribution
    label_counts = np.sum(data['y_train'], axis=0)
    label_dist = [(valid_tags[i], count) for i, count in enumerate(label_counts)]
    label_dist.sort(key=lambda x: x[1], reverse=True)
    
    print("\nLabel Distribution (Top 20):")
    for tag, count in label_dist[:20]:
        print(f"{tag}: {count} ({count/len(data['y_train'])*100:.2f}%)")
    
    # Visualize label distribution
    plt.figure(figsize=(12, 6))
    top_tags = [tag for tag, _ in label_dist[:20]]
    top_counts = [count for _, count in label_dist[:20]]
    plt.bar(top_tags, top_counts)
    plt.title('Top 20 Tags Distribution')
    plt.xlabel('Tags')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    tags_dist_path = os.path.join(BILSTM_3TAGS_DIR, 'top_tags_distribution.png')
    plt.savefig(tags_dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Top tags distribution chart saved to: {tags_dist_path}")
    
    # Calculate class imbalance weights
    pos_weights = torch.tensor(
        [len(data['y_train']) / max(1, count) for count in label_counts],
        dtype=torch.float
    )
    
    # Create datasets
    train_dataset = MovieDataset(data['X_train'], data['y_train'], word_to_idx)
    val_dataset = MovieDataset(data['X_val'], data['y_val'], word_to_idx)
    test_dataset = MovieDataset(data['X_test'], data['y_test'], word_to_idx)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    # Set model parameters
    vocab_size = len(vocab)
    output_dim = len(valid_tags)
    
    # Create model
    model = BiLSTMWithAttention(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, output_dim, N_LAYERS, DROPOUT)
    model = model.to(device)
    
    # Print model structure
    print("\nModel Structure:")
    print(model)
    
    # Calculate number of model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Set loss function and optimizer
    # Use weighted BCE loss to handle class imbalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    
    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Save model configuration
    model_config = {
        'vocab_size': vocab_size,
        'embedding_dim': EMBEDDING_DIM,
        'hidden_dim': HIDDEN_DIM,
        'output_dim': output_dim,
        'n_layers': N_LAYERS,
        'dropout': DROPOUT,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'batch_size': BATCH_SIZE,
        'max_seq_length': MAX_SEQ_LENGTH,
        'max_epochs': MAX_EPOCHS,
        'total_params': total_params,
        'trainable_params': trainable_params,
    }
    
    model_config_path = os.path.join(BILSTM_3TAGS_DIR, 'model_config.json')
    with open(model_config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"Model configuration saved to: {model_config_path}")
    
    # Train model
    model, best_val_f1, training_metrics = train(model, train_loader, val_loader, criterion, optimizer)
    
    # Evaluate validation set
    print("\nEvaluating model on validation set...")
    val_evaluation, (val_labels, val_preds) = evaluate(model, val_loader, criterion, valid_tags, device)
    
    # Display multilabel evaluation metrics for validation set
    print("\nValidation Set Multilabel Evaluation Metrics:")
    print(f"Micro-F1 Score: {val_evaluation['multilabel_metrics']['micro_f1_score']:.4f} - Treats all label instances as a whole")
    print(f"Tag Recall: {val_evaluation['multilabel_metrics']['tag_recall']:.4f} - Proportion of tags correctly predicted by the model")
    print(f"Tags Learned: {val_evaluation['multilabel_metrics']['tags_learned']}/{val_evaluation['multilabel_metrics']['total_tags_used']} - Number of tags the model can correctly predict")
    
    # Evaluate test set
    print("\nEvaluating model on test set...")
    test_evaluation, (test_labels, test_preds) = evaluate(model, test_loader, criterion, valid_tags, device)
    
    # Display multilabel evaluation metrics for test set
    print("\nTest Set Multilabel Evaluation Metrics:")
    print(f"Micro-F1 Score: {test_evaluation['multilabel_metrics']['micro_f1_score']:.4f} - Treats all label instances as a whole")
    print(f"Tag Recall: {test_evaluation['multilabel_metrics']['tag_recall']:.4f} - Proportion of tags correctly predicted by the model")
    print(f"Tags Learned: {test_evaluation['multilabel_metrics']['tags_learned']}/{test_evaluation['multilabel_metrics']['total_tags_used']} - Number of tags the model can correctly predict")
    
    # Plot tag performance
    top_tags_path = os.path.join(BILSTM_3TAGS_DIR, 'top_tags_performance.png')
    plot_top_tags_performance(val_evaluation, save_path=top_tags_path)
    
    # Plot new evaluation metrics
    new_metrics_plot_path = os.path.join(BILSTM_3TAGS_DIR, 'evaluation_metrics.png')
    plot_new_metrics(
        val_evaluation['multilabel_metrics'], 
        test_evaluation['multilabel_metrics'],
        new_metrics_plot_path
    )
    
    # Save best model and results
    best_checkpoint = torch.load(BILSTM_BEST_MODEL_PATH)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Save model and results
    results = {
        'model_type': 'bilstm_3tags',
        'best_validation': val_evaluation,
        'test': test_evaluation,
        'training_metrics': training_metrics,
        'multilabel_metrics_comparison': {
            'best_validation': val_evaluation['multilabel_metrics'],
            'test': test_evaluation['multilabel_metrics']
        },
        'model_config': model_config,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save as JSON
    results_path = os.path.join(BILSTM_3TAGS_DIR, 'results.json')
    with open(results_path, 'w') as f:
        # Handle numpy types
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nBiLSTM model training and evaluation complete! All results saved to {BILSTM_3TAGS_DIR}")
    print(f"Best model: {BILSTM_BEST_MODEL_PATH}")
    print(f"Results JSON: {results_path}")
    
    # Calculate total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    return model, results

if __name__ == "__main__":
    main() 
  
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
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import re

# 导入配置文件中的设置
from config import (
    DATA_DIR, OUTPUT_DIR,
    VALID_TAGS_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH,
    BILSTM_MODEL_PATH, BILSTM_VOCAB_PATH, BILSTM_RESULTS_PATH,
    TEXT_LENGTH_DIST_PATH, BILSTM_TOP_TAGS_PERFORMANCE_PATH, BILSTM_LOSS_CURVE_PATH,
    SEED, MAX_VOCAB_SIZE, MAX_SEQ_LENGTH, BATCH_SIZE, 
    EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, 
    LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE
)

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    
    # 用于早停
    best_val_f1 = 0
    no_improvement = 0
    best_model = None
    
    # 保存损失和F1分数的历史
    train_losses = []
    val_losses = []
    val_f1s = []
    
    # 训练循环
    for epoch in range(1, n_epochs + 1):
        # 训练模式
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")
        
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
            
            # 更新进度条
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 评估验证集
        model.eval()
        val_loss = 0
        
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batch in val_loader:
                texts, labels = batch
                texts = texts.to(device)
                labels = labels.to(device)
                
                predictions = model(texts)
                loss = criterion(predictions, labels)
                val_loss += loss.item()
                
                # 应用sigmoid并转换为二值预测
                preds = torch.sigmoid(predictions) >= 0.5
                
                # 将预测和标签添加到列表
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
        
        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 计算F1分数
        val_labels = np.vstack(all_labels)
        val_preds = np.vstack(all_preds)
        
        # 样本级别F1分数
        sample_f1 = f1_score(val_labels, val_preds, average='samples', zero_division=0)
        val_f1s.append(sample_f1)
        
        # 打印进度
        print(f'Epoch {epoch}/{n_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Val F1: {sample_f1:.4f}')
        
        # 检查是否有改进
        if sample_f1 > best_val_f1:
            best_val_f1 = sample_f1
            no_improvement = 0
            best_model = model.state_dict().copy()
            print(f'  新的最佳F1分数！')
        else:
            no_improvement += 1
            print(f'  没有改进: {no_improvement}/{patience}')
            
            if no_improvement >= patience:
                print('早停: 验证F1分数已经连续多个epoch没有改进')
                break
    
    # 加载最佳模型
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_f1s, label='Val F1')
    plt.legend()
    plt.title('F1 Score')
    
    plt.tight_layout()
    plt.savefig(BILSTM_LOSS_CURVE_PATH)
    plt.close()
    
    return model, best_val_f1

def evaluate(model, data_loader, valid_tags):
    """评估模型，计算论文中使用的评估指标"""
    model.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for texts, labels in tqdm(data_loader, desc="评估模型"):
            texts = texts.to(device)
            labels = labels.to(device)
            
            # 前向传播
            predictions = model(texts)
            
            # 转换为二值预测
            preds = (torch.sigmoid(predictions) > 0.5).float()
            
            # 存储预测和真实值
            y_pred.append(preds.cpu().numpy())
            y_true.append(labels.cpu().numpy())
    
    # 合并批次的预测和真实值
    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)
    
    # 计算总体指标
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"\n总体性能指标:")
    print(f"Micro-平均 - 精确率: {precision_micro:.4f}, 召回率: {recall_micro:.4f}, F1: {f1_micro:.4f}")
    print(f"Macro-平均 - 精确率: {precision_macro:.4f}, 召回率: {recall_macro:.4f}, F1: {f1_macro:.4f}")
    
    # 计算每个标签的F1分数
    f1_scores = []
    for i, tag in enumerate(valid_tags):
        tag_precision = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        tag_recall = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        tag_f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1_scores.append((tag, tag_precision, tag_recall, tag_f1))
    
    # 按F1分数排序
    f1_scores.sort(key=lambda x: x[3], reverse=True)
    
    # 打印每个标签的F1分数
    print("\n各标签的性能指标 (前20个):")
    print(f"{'标签':<20} {'精确率':<10} {'召回率':<10} {'F1':<10}")
    print("-" * 50)
    for tag, tag_precision, tag_recall, tag_f1 in f1_scores[:20]:
        print(f"{tag:<20} {tag_precision:<10.4f} {tag_recall:<10.4f} {tag_f1:<10.4f}")
    
    # 保存评估结果
    evaluation = {
        'precision_micro': float(precision_micro),
        'recall_micro': float(recall_micro),
        'f1_micro': float(f1_micro),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'tag_metrics': {tag: {'precision': float(p), 'recall': float(r), 'f1': float(f)} for tag, p, r, f in f1_scores}
    }
    
    return evaluation, y_pred

def plot_top_tags_performance(evaluation_results, top_n=20):
    """绘制标签性能柱状图"""
    # 提取标签F1分数
    tag_scores = evaluation_results['tag_f1_scores']
    
    # 按F1分数排序
    sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 选择前N个标签
    top_tags = sorted_tags[:top_n]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制柱状图
    tags, scores = zip(*top_tags)
    ax.bar(tags, scores, color='skyblue')
    
    # 设置标题和标签
    ax.set_title(f'Top {top_n} Tags Performance (F1 Score)', fontsize=15)
    ax.set_xlabel('Tags', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    
    # 旋转x轴标签
    plt.xticks(rotation=45, ha='right')
    
    # 为每个条形添加数值标签
    for i, score in enumerate(scores):
        ax.text(i, score + 0.01, f'{score:.2f}', ha='center', fontsize=8)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(BILSTM_TOP_TAGS_PERFORMANCE_PATH)
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

def main():
    """主函数"""
    print("="*80)
    print("开始训练BiLSTM电影类型分类模型 (基于论文实现)")
    print("="*80)
    
    # 加载数据
    data = load_data()
    
    # 构建词汇表
    vocab, word_to_idx = build_vocabulary(data['X_train'])
    
    # 统计标签分布
    label_counts = np.sum(data['y_train'], axis=0)
    label_dist = [(data['valid_tags'][i], count) for i, count in enumerate(label_counts)]
    label_dist.sort(key=lambda x: x[1], reverse=True)
    
    print("\n标签分布 (前20个):")
    for tag, count in label_dist[:20]:
        print(f"{tag}: {count} ({count/len(data['y_train'])*100:.2f}%)")
    
    # 计算类别不平衡权重
    pos_weights = torch.tensor(
        [len(data['y_train']) / max(1, count) for count in label_counts],
        dtype=torch.float
    )
    
    # 创建数据集
    train_dataset = MovieDataset(data['X_train'], data['y_train'], word_to_idx)
    val_dataset = MovieDataset(data['X_val'], data['y_val'], word_to_idx)
    test_dataset = MovieDataset(data['X_test'], data['y_test'], word_to_idx)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    # 设置模型参数
    vocab_size = len(vocab)
    output_dim = len(data['valid_tags'])
    
    # 创建模型
    model = BiLSTMWithAttention(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, output_dim, N_LAYERS, DROPOUT)
    model = model.to(device)
    
    # 打印模型结构
    print("\n模型结构:")
    print(model)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 设置损失函数和优化器
    # 使用带权重的BCE损失函数来处理类别不平衡
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 训练模型
    model, best_val_f1 = train(model, train_loader, val_loader, criterion, optimizer)
    
    # 评估验证集
    print("\n在验证集上评估模型...")
    val_evaluation, _ = evaluate(model, val_loader, data['valid_tags'])
    
    # 评估测试集
    print("\n在测试集上评估模型...")
    test_evaluation, _ = evaluate(model, test_loader, data['valid_tags'])
    
    # 绘制标签性能
    plot_top_tags_performance(val_evaluation)
    
    # 保存模型和结果
    save_model_and_results(model, word_to_idx, val_evaluation, test_evaluation)
    
    print("\nBiLSTM模型训练和评估完成!")

if __name__ == "__main__":
    main() 
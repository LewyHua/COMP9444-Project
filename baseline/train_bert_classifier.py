#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BERT模型训练脚本，用于电影类型分类
"""

import os
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl  # 添加缺少的matplotlib导入
import seaborn as sns
from tqdm import tqdm
from math import ceil
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn import BCEWithLogitsLoss

from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup, logging, AutoModel, AutoConfig, AutoTokenizer

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from config import (
    BATCH_SIZE, MAX_EPOCHS, SEED, LEARNING_RATE, WEIGHT_DECAY, 
    BERT_MODEL_NAME, LOCAL_BERT_MODEL_PATH, OFFLINE_MODE, PATIENCE,
    BERT_BEST_MODEL_PATH, BERT_WORST_MODEL_PATH, BERT_TRAINING_METRICS_PATH,
    BERT_3TAGS_DIR, BERT_3TAGS_RESULTS_PATH, BERT_3TAGS_CONFUSION_MATRIX_PATH,
    VALID_TAGS_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH,
    ALL_DATA_PATH as MOVIE_DATA_PATH,  # 使用config中已有的ALL_DATA_PATH作为MOVIE_DATA_PATH
    BERT_MAX_LENGTH as MAX_LENGTH,  # 使用config中已有的BERT_MAX_LENGTH作为MAX_LENGTH
    BERT_LEARNING_RATE, CLASSIFIER_LEARNING_RATE, BERT_DROPOUT,
    CPU_BATCH_SIZE,
    BERT_3TAGS_LOSS_CURVE_PATH, BERT_3TAGS_TAG_CORR_PATH, BERT_3TAGS_PERFORMANCE_DIST_PATH,
    BERT_3TAGS_TOP_TAGS_PATH, BERT_3TAGS_EPOCH_METRICS_PATH,
    BERT_3TAGS_TAG_HEATMAP_PATH, BERT_TEXT_LENGTH_DIST_PATH,
    BERT_TOP_TAGS_DIST_PATH, BERT_SAMPLE_PRECISION_DIST_PATH,
    BERT_TAG_FREQ_COMPARISON_PATH, BERT_MODEL_CONFIG_PATH,
    OUTPUT_DIR, PREDICTION_THRESHOLD  # 添加预测阈值参数
)

# 其他常量定义
NUM_WORKERS = 4   # 数据加载器工作线程数
WARMUP_RATIO = 0.1  # 预热比例

# 设置matplotlib不使用中文字体，防止警告
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.unicode_minus'] = False
# 强制使用特定的字体文件
plt.rcParams['svg.fonttype'] = 'none'
print("CUDA available: ", torch.cuda.is_available())

# 设置随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BERT_3TAGS_DIR, exist_ok=True)

class MovieDataset(Dataset):
    def __init__(self, input_ids, attention_mask, token_type_ids, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'token_type_ids': self.token_type_ids[idx],
            'labels': self.labels[idx]
        }

class BertMovieClassifier(nn.Module):
    """基于BERT的电影分类器"""
    
    def __init__(self, bert_model_name, num_labels, dropout_rate=BERT_DROPOUT):
        super(BertMovieClassifier, self).__init__()
        
        # 加载BERT模型
        try:
            self.bert = AutoModel.from_pretrained(bert_model_name)
            print(f"已成功加载BERT模型: {bert_model_name}")
        except Exception as e:
            print(f"BERT模型加载失败: {e}")
            # 创建一个随机初始化的模型作为替代
            config = AutoConfig.from_pretrained(bert_model_name)
            self.bert = AutoModel.from_config(config)
            print("使用随机初始化的BERT模型")
        
        # BERT输出维度 (通常为768)
        self.bert_output_dim = self.bert.config.hidden_size
        
        # 分类器层
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_labels)
        )
        
        print(f"初始化BERT分类器: {bert_model_name}, 输出维度: {self.bert_output_dim}")
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        # 获取BERT输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 使用CLS标记的最后隐藏状态
        pooled_output = outputs.pooler_output
        
        # 通过分类器
        return self.classifier(pooled_output)

def load_data():
    """加载预处理好的数据集"""
    print("正在加载预处理好的数据...")
    
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
    
    # 打印标签分布
    print("\n标签分布 (前20个):")
    label_counts = np.sum(y_train, axis=0)
    label_dist = [(valid_tags[i], count) for i, count in enumerate(label_counts)]
    label_dist.sort(key=lambda x: x[1], reverse=True)
    
    for tag, count in label_dist[:20]:
        print(f"{tag}: {count} ({count/len(y_train)*100:.2f}%)")
    
    # 统计文本长度
    text_lengths = [len(str(text).split()) for text in X_train]
    avg_length = np.mean(text_lengths)
    median_length = np.median(text_lengths)
    print(f"平均文本长度: {avg_length:.2f} 词")
    print(f"中位文本长度: {median_length:.2f} 词")
    
    # 返回处理好的数据
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'valid_tags': valid_tags,
        'tag_columns': tag_columns
    }

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 设置cudnn为确定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"已设置随机种子: {seed}")

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, n_epochs=MAX_EPOCHS, patience=PATIENCE):
    """训练BERT模型"""
    # 设置随机种子确保可重复性
    set_seed(SEED)
    
    print(f"开始训练，总共 {n_epochs} 个epoch...")
    
    # 初始化训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'learning_rates': []
    }
    
    # 用于早停的变量
    best_val_f1 = 0
    best_model = None
    no_improvement = 0
    
    # 记录开始时间
    start_time = time.time()
    
    # 训练循环
    for epoch in range(n_epochs):
        # 记录开始时间
        epoch_start = time.time()
        
        print(f"\n------ Epoch {epoch+1}/{n_epochs} ------")
        
        # 训练阶段
        model.train()
        train_loss = 0
        
        # 使用tqdm进度条
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # 将数据移到GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device) if batch.get('token_type_ids') is not None else None
            labels = batch['labels'].to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 更新参数
            optimizer.step()
            
            # 更新学习率
            scheduler.step()
            
            # 累加损失
            train_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        
        # 评估阶段
        print("\n开始验证...")
        val_metrics = evaluate(model, val_loader, criterion, None)
        
        # 更新训练历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # 打印训练指标
        print(f"训练损失: {avg_train_loss:.4f}")
        print(f"验证损失: {val_metrics['loss']:.4f}")
        print(f"验证准确率: {val_metrics['accuracy']:.4f}")
        print(f"验证F1: {val_metrics['f1']:.4f}")
        
        # 计算epoch耗时
        epoch_time = time.time() - epoch_start
        print(f"Epoch耗时: {epoch_time:.2f}秒")
        
        # 检查是否有改进
        current_f1 = val_metrics['f1']
        if current_f1 > best_val_f1:
            print(f"验证F1从 {best_val_f1:.4f} 提升到 {current_f1:.4f}")
            best_val_f1 = current_f1
            best_model = copy.deepcopy(model.state_dict())
            no_improvement = 0
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1': best_val_f1,
                'history': history
            }, BERT_BEST_MODEL_PATH)
            print(f"保存最佳模型到 {BERT_BEST_MODEL_PATH}")
        else:
            no_improvement += 1
            print(f"验证F1没有提升，当前 {current_f1:.4f}，最佳 {best_val_f1:.4f}")
            print(f"无提升次数: {no_improvement}/{patience}")
        
        # 检查是否需要早停
        if no_improvement >= patience:
            print(f"\n连续 {patience} 个epoch没有改进，提前停止训练。")
            break
    
    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"\n训练完成，总耗时: {total_time:.2f} 秒")
    
    # 加载最佳模型
    if best_model is not None:
        model.load_state_dict(best_model)
        print(f"已加载最佳模型 (F1={best_val_f1:.4f})")
    
    # 添加总训练时间到历史记录
    history['total_training_time'] = total_time
    history['best_val_f1'] = best_val_f1
    
    return history

def evaluate(model, data_loader, criterion, label_encoder):
    """评估模型性能"""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0
    
    # 定义Top-k和最小概率阈值
    MIN_PROB_THRESHOLD = 0.1  # 最小概率阈值，可调整
    k = 3  # 选择前k个标签
    
    with torch.no_grad():
        for batch in data_loader:
            # 获取批次数据
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device) if batch.get('token_type_ids') is not None else None
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 计算预测概率
            probs = torch.sigmoid(outputs)
            
            # 方法1：使用阈值
            preds_threshold = (probs > PREDICTION_THRESHOLD).float()
            
            # 方法2：Top-k + 最小概率阈值
            # 获取Top-k的索引和概率值
            top_probs, top_indices = torch.topk(probs, k=k, dim=1)
            
            # 初始化预测矩阵
            preds_topk = torch.zeros_like(probs)
            
            # 为每个样本填充符合条件的预测
            for i, (indices, values) in enumerate(zip(top_indices, top_probs)):
                # 只保留概率超过阈值的标签
                valid_mask = values >= MIN_PROB_THRESHOLD
                valid_indices = indices[valid_mask]
                
                # 将合格的标签设为1
                if len(valid_indices) > 0:
                    preds_topk[i, valid_indices] = 1.0
            
            # 使用Top-k + 最小概率阈值方法的预测
            preds = preds_topk
            
            # 收集所有预测和标签
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # 合并所有批次的结果
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)
    
    # 计算评估指标
    precision = precision_score(all_labels, all_preds, average='samples', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='samples', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='samples', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 计算每个标签的指标
    per_label_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_label_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    per_label_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # 计算每个标签的支持度（实际为正的样本数）
    per_label_support = all_labels.sum(axis=0)
    
    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    
    # 汇总结果
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'loss': avg_loss,
        'per_label': {
            'precision': per_label_precision,
            'recall': per_label_recall,
            'f1': per_label_f1,
            'support': per_label_support
        },
        'prediction_method': 'top_k_with_threshold',
        'k_value': k,
        'min_threshold': MIN_PROB_THRESHOLD
    }
    
    # 打印总体评估结果
    print(f"评估结果 (Top-{k} + 最小阈值={MIN_PROB_THRESHOLD}):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Loss: {avg_loss:.4f}")
    
    # 只有在提供了label_encoder的情况下才打印标签性能
    if label_encoder is not None:
        # 打印表现最好和最差的几个标签
        tag_names = label_encoder['tag_names']
        f1_with_labels = [(tag_names[i], f1, per_label_support[i]) for i, f1 in enumerate(per_label_f1)]
        
        # 按F1分数排序，只考虑支持度大于5的标签
        f1_with_labels = [(label, f1, support) for label, f1, support in f1_with_labels if support > 5]
        f1_with_labels.sort(key=lambda x: x[1], reverse=True)
        
        print("\n表现最好的5个标签:")
        for label, f1, support in f1_with_labels[:5]:
            print(f"{label}: F1={f1:.4f}, Support={support}")
        
        print("\n表现最差的5个标签:")
        for label, f1, support in f1_with_labels[-5:]:
            print(f"{label}: F1={f1:.4f}, Support={support}")
    
    return metrics

def plot_top_tags_performance(evaluation_results, top_n=20):
    """绘制标签性能柱状图，同时展示传统和Top3评估结果"""
    # Extract Tag F1 Scores
    tag_metrics = evaluation_results['tag_metrics']
    
    # Prepare Data
    tags = []
    f1_traditional = []
    f1_top3 = []
    
    # Sort by Top3 F1 Score
    sorted_tags = sorted(
        tag_metrics.items(), 
        key=lambda x: x[1]['top3']['f1'], 
        reverse=True
    )
    
    # Select Top N Tags
    for tag, metrics in sorted_tags[:top_n]:
        tags.append(tag)
        f1_traditional.append(metrics['traditional']['f1'])
        f1_top3.append(metrics['top3']['f1'])
    
    # Create Chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set x-axis Position
    x = np.arange(len(tags))
    width = 0.35
    
    # Plot Bar Chart
    ax.bar(x - width/2, f1_traditional, width, label='Traditional F1')
    ax.bar(x + width/2, f1_top3, width, label='F1@3')
    
    # Set Title and Labels
    ax.set_title(f'Top {top_n} Tags Performance', fontsize=15)
    ax.set_xlabel('Tags', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    ax.legend()
    
    # Rotate x-axis Labels
    plt.xticks(rotation=45, ha='right')
    
    # Adjust Layout
    plt.tight_layout()
    
    # Save Chart
    plt.savefig(BERT_3TAGS_TOP_TAGS_PATH)
    plt.close()

def create_dataset(train_df, val_df, test_df, tokenizer, label_encoder):
    """为BERT模型创建数据集"""
    print("创建BERT数据集...")
    
    if tokenizer is None:
        raise ValueError("分词器为空，无法创建数据集")
    
    def process_df(df, tag_columns):
        # 从DataFrame中提取文本和标签
        texts = df['plot_synopsis'].fillna('').values
        # 标签需要输出为numpy数组，确保是浮点型
        labels = df[tag_columns].values.astype(np.float32)
        
        # 分词并转换为张量
        encoding = tokenizer(
            list(texts),
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # 创建数据集
        dataset = MovieDataset(
            encoding['input_ids'],
            encoding['attention_mask'],
            encoding['token_type_ids'] if 'token_type_ids' in encoding else None,
            torch.tensor(labels, dtype=torch.float32)
        )
        
        return dataset
    
    # 获取标签列名
    tag_columns = [f"tag_{tag.replace(' ', '_')}" for tag in label_encoder['tag_names']]
    
    # 处理三个数据集
    train_dataset = process_df(train_df, tag_columns)
    val_dataset = process_df(val_df, tag_columns)
    test_dataset = process_df(test_df, tag_columns)
    
    print(f"数据集创建成功:")
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    
    return train_dataset, val_dataset, test_dataset

def create_optimizer_and_scheduler(model, num_training_steps):
    """创建优化器和学习率调度器"""
    # 设置不同的学习率 - BERT层使用较小的学习率
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': BERT_LEARNING_RATE},
        {'params': [p for n, p in model.named_parameters() if 'bert' not in n], 'lr': CLASSIFIER_LEARNING_RATE}
    ]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters)
    
    # 计算总的训练步数
    total_steps = num_training_steps * MAX_EPOCHS
    
    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps
    )
    
    return optimizer, scheduler

def visualize_training_history(training_history):
    """可视化训练历史"""
    epoch_nums = training_history['epoch_nums']
    train_losses = training_history['train_losses']
    val_losses = training_history['val_losses']
    train_f1s = training_history['train_f1s']
    val_f1s = training_history['val_f1s']
    
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
    plt.savefig(BERT_3TAGS_LOSS_CURVE_PATH)
    plt.close()
    
def visualize_performance(evaluation, valid_tags):
    """可视化模型性能"""
    # 获取标签级别性能
    label_perf = evaluation['label_performance']
    
    # 按F1分数排序
    sorted_perf = sorted([(i, perf['f1']) for i, perf in enumerate(label_perf)], 
                       key=lambda x: x[1], reverse=True)
    
    # 选择前15个和后15个标签
    top_indices = [idx for idx, _ in sorted_perf[:15]]
    bottom_indices = [idx for idx, _ in sorted_perf[-15:]]
    selected_indices = top_indices + bottom_indices
    
    # 提取标签名称和F1分数
    selected_tags = [valid_tags[i] for i in selected_indices]
    selected_f1 = [label_perf[i]['f1'] for i in selected_indices]
    
    # 创建颜色映射 - 顶部标签为绿色，底部标签为红色
    colors = ['green'] * 15 + ['red'] * 15
    
    # 绘制柱状图
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(selected_tags)), selected_f1, color=colors)
    plt.xticks(range(len(selected_tags)), selected_tags, rotation=45, ha='right')
    plt.xlabel('Tags')
    plt.ylabel('F1 Score')
    plt.title('Top 15 and Bottom 15 Tags by F1 Score')
    plt.tight_layout()
    plt.savefig(BERT_3TAGS_TOP_TAGS_PATH)
    plt.close()

def create_tokenizer():
    """创建BERT分词器"""
    print("正在初始化BERT分词器...")
    
    try:
        # 尝试从Hugging Face加载
        if OFFLINE_MODE:
            # 从本地加载
            tokenizer = AutoTokenizer.from_pretrained(LOCAL_BERT_MODEL_PATH)
            print(f"已从本地路径加载BERT分词器: {LOCAL_BERT_MODEL_PATH}")
        else:
            # 从Hugging Face在线加载
            tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
            print(f"已从Hugging Face加载BERT分词器: {BERT_MODEL_NAME}")
        
        return tokenizer
    except Exception as e:
        print(f"无法加载BERT分词器: {e}")
        # 如果加载失败，返回None
        return None

def prepare_data(X_train, y_train, X_val, y_val, X_test, y_test, valid_tags, tag_columns):
    """从预处理的数据准备训练所需格式"""
    # 创建DataFrame用于训练和测试
    train_df = pd.DataFrame({'text': X_train})
    val_df = pd.DataFrame({'text': X_val})
    test_df = pd.DataFrame({'text': X_test})
    
    # 将标签值转换为DataFrame的一部分
    for i, tag in enumerate(valid_tags):
        train_df[tag] = y_train[:, i]
        val_df[tag] = y_val[:, i]
        test_df[tag] = y_test[:, i]
    
    # 创建与LabelBinarizer等价的标签编码器
    label_encoder = {
        'classes_': np.array(valid_tags)
    }
    
    print(f"数据准备完成:")
    print(f"训练集: {len(train_df)} 样本")
    print(f"验证集: {len(val_df)} 样本")
    print(f"测试集: {len(test_df)} 样本")
    print(f"标签数量: {len(valid_tags)}")
    
    return train_df, val_df, test_df, label_encoder

def encode_labels(tags, label_encoder):
    """将标签编码为二进制向量"""
    # 使用LabelBinarizer直接转换标签
    encoded = np.zeros(len(label_encoder['classes_']))
    
    # 为每个存在的标签设置对应位置为1
    for tag in tags:
        if tag in label_encoder['classes_']:
            idx = list(label_encoder['classes_']).index(tag)
            encoded[idx] = 1
    
    return encoded

def print_dataset_stats(train_df, label_encoder):
    """打印数据集统计信息"""
    print("\n标签分布 (前20个):")
    
    # 获取标签列名
    label_names = label_encoder['classes_']
    
    # 统计标签频率
    tag_counts = {}
    for tag in label_names:
        tag_counts[tag] = train_df[tag].sum()
    
    # 按频率降序排序
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    
    # 打印前20个标签的频率
    for tag, count in sorted_tags[:20]:
        percentage = (count / len(train_df)) * 100
        print(f"{tag}: {int(count)} ({percentage:.2f}%)")
    
    # 文本长度统计
    text_lengths = train_df['text'].apply(lambda x: len(str(x).split()))
    print(f"平均文本长度: {text_lengths.mean():.2f} 词")
    print(f"中位文本长度: {text_lengths.median():.2f} 词")
    print(f"标签数量: {len(label_encoder['classes_'])}")

def save_training_history(history):
    """保存训练历史数据"""
    with open(os.path.join(BERT_3TAGS_DIR, 'training_history.json'), 'w') as f:
        # 将numpy数组转换成列表
        history_serializable = {}
        for k, v in history.items():
            if isinstance(v, (np.ndarray, np.float32, np.float64, np.int64)):
                history_serializable[k] = v.tolist() if hasattr(v, 'tolist') else float(v)
            elif isinstance(v, list):
                history_serializable[k] = [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in v]
            else:
                history_serializable[k] = v
        json.dump(history_serializable, f, indent=2)
    print(f"训练历史已保存到 {os.path.join(BERT_3TAGS_DIR, 'training_history.json')}")

def save_test_results(metrics, confusion, label_encoder):
    """保存测试结果"""
    # 保存测试指标
    with open(os.path.join(BERT_3TAGS_DIR, 'test_metrics.json'), 'w') as f:
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                serializable_metrics[k] = {}
                for k2, v2 in v.items():
                    if isinstance(v2, (np.ndarray, np.float32, np.float64)):
                        serializable_metrics[k][k2] = v2.tolist() if hasattr(v2, 'tolist') else float(v2)
                    else:
                        serializable_metrics[k][k2] = v2
            elif isinstance(v, (np.ndarray, np.float32, np.float64)):
                serializable_metrics[k] = v.tolist() if hasattr(v, 'tolist') else float(v)
            else:
                serializable_metrics[k] = v
        json.dump(serializable_metrics, f, indent=2)
    print(f"测试结果已保存到 {os.path.join(BERT_3TAGS_DIR, 'test_metrics.json')}")

def save_model(model, filename):
    """保存模型"""
    torch.save(model.state_dict(), filename)
    print(f"模型已保存到 {filename}")

def generate_error_analysis(model, test_loader, test_df, label_encoder):
    """生成错误分析结果"""
    model.eval()
    errors = []
    
    # 定义Top-k和最小概率阈值，与evaluate函数保持一致
    MIN_PROB_THRESHOLD = 0.1  # 最小概率阈值，可调整
    k = 3  # 选择前k个标签
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # 获取批次数据
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device) if batch.get('token_type_ids') is not None else None
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # 计算预测值
            probs = torch.sigmoid(outputs)
            
            # 使用Top-k + 最小概率阈值方法
            # 获取Top-k的索引和概率值
            top_probs, top_indices = torch.topk(probs, k=k, dim=1)
            
            # 初始化预测矩阵
            preds = torch.zeros_like(probs)
            
            # 为每个样本填充符合条件的预测
            for j, (indices, values) in enumerate(zip(top_indices, top_probs)):
                # 只保留概率超过阈值的标签
                valid_mask = values >= MIN_PROB_THRESHOLD
                valid_indices = indices[valid_mask]
                
                # 将合格的标签设为1
                if len(valid_indices) > 0:
                    preds[j, valid_indices] = 1.0
            
            # 找出预测错误的样本
            for j in range(len(preds)):
                idx = i * test_loader.batch_size + j
                if idx >= len(test_df):
                    break
                    
                sample_preds = preds[j].cpu().numpy()
                sample_labels = labels[j].cpu().numpy()
                sample_probs = probs[j].cpu().numpy()
                
                # 计算错误
                false_positives = np.where((sample_preds == 1) & (sample_labels == 0))[0]
                false_negatives = np.where((sample_preds == 0) & (sample_labels == 1))[0]
                
                if len(false_positives) > 0 or len(false_negatives) > 0:
                    # 获取样本文本
                    text = test_df.iloc[idx]['plot_synopsis']
                    
                    # 获取实际标签
                    actual_tags = []
                    for k, tag in enumerate(label_encoder['tag_names']):
                        if sample_labels[k] == 1:
                            actual_tags.append(tag)
                    
                    # 获取错误预测的标签
                    fp_tags = [label_encoder['tag_names'][idx] for idx in false_positives]
                    fn_tags = [label_encoder['tag_names'][idx] for idx in false_negatives]
                    
                    errors.append({
                        'text': text[:200] + '...' if len(text) > 200 else text,
                        'actual_tags': actual_tags,
                        'false_positives': fp_tags,
                        'false_negatives': fn_tags,
                        'probabilities': {
                            'fp': {label: float(sample_probs[idx]) for idx, label in zip(false_positives, fp_tags)},
                            'fn': {label: float(sample_probs[idx]) for idx, label in zip(false_negatives, fn_tags)}
                        }
                    })
    
    return errors

def save_error_analysis(errors):
    """保存错误分析结果"""
    # 只保存前100个错误案例以节省空间
    with open(os.path.join(BERT_3TAGS_DIR, 'error_analysis.json'), 'w') as f:
        json.dump(errors[:100], f, indent=2)
    print(f"错误分析结果已保存到 {os.path.join(BERT_3TAGS_DIR, 'error_analysis.json')}")

def main():
    """主函数"""
    print("="*80)
    print("开始训练BERT电影标签分类模型")
    print("="*80)
    print(f"训练结果将保存到: {BERT_3TAGS_DIR}")
    print(f"使用随机种子: {SEED}")
    print(f"BERT模型: {BERT_MODEL_NAME if not OFFLINE_MODE else LOCAL_BERT_MODEL_PATH}")
    
    # 加载数据
    data = load_data()
    
    # 创建标签编码器
    label_encoder = {
        'tag_names': data['valid_tags'],
        'tag_columns': data['tag_columns']
    }
    
    # 创建tokenizer
    bert_tokenizer = create_tokenizer()
    if bert_tokenizer is None:
        print("无法创建BERT分词器，退出训练")
        return None
    
    # 准备数据集
    train_dataset, val_dataset, test_dataset = create_dataset(
        pd.read_csv(TRAIN_DATA_PATH),
        pd.read_csv(VAL_DATA_PATH),
        pd.read_csv(TEST_DATA_PATH),
        bert_tokenizer,
        label_encoder
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 创建BERT模型
    print("\n创建BERT模型...")
    bert_model = BertMovieClassifier(
        BERT_MODEL_NAME if not OFFLINE_MODE else LOCAL_BERT_MODEL_PATH,
        len(data['valid_tags'])
    ).to(device)
    
    # 创建优化器和学习率调度器
    optimizer, scheduler = create_optimizer_and_scheduler(
        bert_model, 
        len(train_loader) * MAX_EPOCHS
    )
    
    # 训练BERT模型
    criterion = BCEWithLogitsLoss()
    print("\n开始训练BERT模型...")
    
    training_history = train(
        bert_model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scheduler,
        n_epochs=MAX_EPOCHS
    )
    
    # 保存训练历史
    save_training_history(training_history)
    
    # 评估BERT模型
    print("\n在验证集上评估BERT模型...")
    val_metrics = evaluate(bert_model, val_loader, criterion, label_encoder)
    print("\n在测试集上评估BERT模型...")
    test_metrics = evaluate(bert_model, test_loader, criterion, label_encoder)
    
    # 保存结果
    print("\n保存评估结果...")
    save_test_results(test_metrics, None, label_encoder)
    
    # 生成错误分析
    error_examples = generate_error_analysis(bert_model, test_loader, 
                                          pd.read_csv(TEST_DATA_PATH), 
                                          label_encoder)
    save_error_analysis(error_examples)
    
    # 保存最终模型
    save_model(bert_model, BERT_BEST_MODEL_PATH)
    
    print(f"\nBERT模型训练和评估完成! 所有结果已保存到 {BERT_3TAGS_DIR}")
    
    return bert_model

if __name__ == "__main__":
    main()

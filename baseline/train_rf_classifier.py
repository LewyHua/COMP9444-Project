#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
电影类型分类 - 基于概率随机森林
使用与BiLSTM相同的数据集和评估方法
"""

import os
import json
import numpy as np
import pandas as pd
import time
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import multilabel_confusion_matrix

# 设置matplotlib不使用中文字体，防止警告
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.unicode_minus'] = False
# 强制使用特定的字体文件
plt.rcParams['svg.fonttype'] = 'none'

# 导入配置文件中的设置
from config import (
    DATA_DIR, 
    VALID_TAGS_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH,
    TFIDF_VECTORIZER_PATH, X_TRAIN_TFIDF_PATH, X_VAL_TFIDF_PATH, X_TEST_TFIDF_PATH,
    SEED, MAX_EPOCHS,
    PREDICTION_THRESHOLD  # 添加预测阈值
)

# 创建新的输出目录
RF_3TAGS_DIR = 'D:/share/base/rf-3tags'
os.makedirs(RF_3TAGS_DIR, exist_ok=True)

# 定义输出路径
RF_MODEL_PATH = os.path.join(RF_3TAGS_DIR, 'rf_model.pkl')
RF_RESULTS_PATH = os.path.join(RF_3TAGS_DIR, 'results.json')
RF_PERF_DIST_PATH = os.path.join(RF_3TAGS_DIR, 'performance_distribution.png')
RF_TAG_HEATMAP_PATH = os.path.join(RF_3TAGS_DIR, 'tag_heatmap.png')
RF_CONFUSION_PATH = os.path.join(RF_3TAGS_DIR, 'confusion_matrix.png')
RF_TOP_TAGS_PATH = os.path.join(RF_3TAGS_DIR, 'top_tags_performance.png')

# 随机种子，保持与BiLSTM一致
np.random.seed(SEED)

def load_data():
    """加载预处理好的TF-IDF特征和标签"""
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
    
    # 加载TF-IDF特征
    X_train = sparse.load_npz(X_TRAIN_TFIDF_PATH)
    X_val = sparse.load_npz(X_VAL_TFIDF_PATH)
    X_test = sparse.load_npz(X_TEST_TFIDF_PATH)
    
    # 提取标签
    y_train = train_df[tag_columns].values
    y_val = val_df[tag_columns].values
    y_test = test_df[tag_columns].values
    
    print(f"数据加载完成:")
    print(f"训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    print(f"标签数量: {len(valid_tags)}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'valid_tags': valid_tags,
        'tag_columns': tag_columns
    }

def train_rf_model(X_train, y_train):
    """训练概率随机森林模型"""
    print("\n开始训练概率随机森林模型...")
    start_time = time.time()
    
    # 配置随机森林模型
    rf_params = {
        'n_estimators': 100,  # 树的数量
        'max_depth': 20,      # 树的最大深度
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'n_jobs': -1,         # 使用所有CPU
        'random_state': SEED,
        'verbose': 1
    }
    
    # 创建基础随机森林模型（使用概率输出）
    base_rf = RandomForestClassifier(**rf_params)
    
    # 使用MultiOutputClassifier包装，处理多标签问题
    model = MultiOutputClassifier(base_rf, n_jobs=1)
    
    # 训练模型
    print("模型训练中，这可能需要一些时间...")
    model.fit(X_train, y_train)
    
    # 计算训练时间
    train_time = time.time() - start_time
    print(f"训练完成，耗时: {train_time:.2f} 秒")
    
    # 保存模型
    with open(RF_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"模型已保存到: {RF_MODEL_PATH}")
    
    return model, train_time

def evaluate_model(model, X, y, valid_tags, mode='验证集'):
    """评估模型性能"""
    # 获取预测概率（MultiOutputClassifier返回的是每个标签的分类器列表）
    y_probs = model.predict_proba(X)
    
    # 转换概率格式
    probs = np.zeros((X.shape[0], len(valid_tags)))
    for i, classifier_probs in enumerate(y_probs):
        probs[:, i] = classifier_probs[:, 1]  # 第1列是正类概率
    
    # 定义Top-k和最小概率阈值
    MIN_PROB_THRESHOLD = 0.1  # 最小概率阈值，可调整
    k = 3  # 选择前k个标签
    
    # 方法1：根据阈值转换为二值预测
    y_pred_threshold = (probs >= PREDICTION_THRESHOLD).astype(int)
    
    # 方法2：Top-k + 最小概率阈值
    y_pred = np.zeros_like(probs)
    
    # 对每个样本应用Top-k和最小阈值
    for i in range(len(y_pred)):
        # 获取概率最高的k个索引
        top_indices = np.argsort(probs[i])[::-1][:k]
        top_probs = probs[i][top_indices]
        
        # 只保留概率超过阈值的标签
        valid_indices = top_indices[top_probs >= MIN_PROB_THRESHOLD]
        
        # 将合格的标签设为1
        if len(valid_indices) > 0:
            y_pred[i, valid_indices] = 1
    
    # 计算指标
    precision = precision_score(y, y_pred, average='samples', zero_division=0)
    recall = recall_score(y, y_pred, average='samples', zero_division=0)
    f1 = f1_score(y, y_pred, average='samples', zero_division=0)
    accuracy = accuracy_score(y, y_pred)
    
    # 计算每个标签的指标
    per_label_precision = precision_score(y, y_pred, average=None, zero_division=0)
    per_label_recall = recall_score(y, y_pred, average=None, zero_division=0)
    per_label_f1 = f1_score(y, y_pred, average=None, zero_division=0)
    
    # 计算每个标签的支持度（即实际为正的样本数）
    per_label_support = y.sum(axis=0)
    
    # 打印总体评估结果
    print(f"\n{mode}评估结果 (Top-{k} + 最小阈值={MIN_PROB_THRESHOLD}):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # 打印表现最好和最差的几个标签
    f1_with_labels = [(valid_tags[i], f1, per_label_support[i]) for i, f1 in enumerate(per_label_f1)]
    
    # 按F1分数排序，只考虑支持度大于5的标签
    f1_with_labels = [(label, f1, support) for label, f1, support in f1_with_labels if support > 5]
    f1_with_labels.sort(key=lambda x: x[1], reverse=True)
    
    print("\n表现最好的5个标签:")
    for label, f1, support in f1_with_labels[:5]:
        print(f"{label}: F1={f1:.4f}, Support={support}")
    
    print("\n表现最差的5个标签:")
    for label, f1, support in f1_with_labels[-5:]:
        print(f"{label}: F1={f1:.4f}, Support={support}")
    
    # 创建结果字典
    evaluation = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'prediction_method': 'top_k_with_threshold',
        'k_value': k,
        'min_threshold': MIN_PROB_THRESHOLD,
        'per_label': {
            valid_tags[i]: {
                'precision': float(per_label_precision[i]),
                'recall': float(per_label_recall[i]),
                'f1': float(per_label_f1[i]),
                'support': int(per_label_support[i])
            } for i in range(len(valid_tags))
        }
    }
    
    return evaluation, y_pred, probs

def main():
    """主函数"""
    print("="*80)
    print("开始训练随机森林电影标签分类模型")
    print("="*80)
    print(f"训练结果将保存到: {RF_3TAGS_DIR}")
    print(f"使用随机种子: {SEED}")
    print(f"从config.py导入的MAX_EPOCHS: {MAX_EPOCHS}")
    
    # 加载数据
    data = load_data()
    
    # 训练模型
    model, train_time = train_rf_model(data['X_train'], data['y_train'])
    
    # 评估验证集
    print("\n在验证集上评估模型...")
    val_evaluation, val_pred, val_probs = evaluate_model(model, data['X_val'], data['y_val'], data['valid_tags'], mode='验证集')
    
    # 评估测试集
    print("\n在测试集上评估模型...")
    test_evaluation, test_pred, test_probs = evaluate_model(model, data['X_test'], data['y_test'], data['valid_tags'], mode='测试集')
    
    # 准备结果
    results = {
        'model_type': 'random_forest',
        'validation': val_evaluation,
        'test': test_evaluation,
        'training_time': train_time,
        'model_params': {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 保存为JSON
    with open(RF_RESULTS_PATH, 'w') as f:
        # 转换numpy值
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    
    print(f"\n随机森林模型训练和评估完成! 所有结果已保存到 {RF_3TAGS_DIR}")
    
    return model, results

if __name__ == "__main__":
    main() 
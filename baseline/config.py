#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MPST数据集处理与模型训练配置文件
此文件包含所有路径设置，可根据实际需求进行修改
"""

import os
import platform

# 数据目录设置
INPUT_DIR = './MPST_v2'  # 原始MPST数据集目录
DATA_DIR = './dataset'   # 处理后数据输出目录
OUTPUT_DIR = './models'  # 模型和结果输出目录

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 文件路径设置
TRAIN_IDS_PATH = os.path.join(INPUT_DIR, 'train_ids.txt')
TEST_IDS_PATH = os.path.join(INPUT_DIR, 'test_ids.txt')
ALL_DATA_PATH = os.path.join(INPUT_DIR, 'all_data.csv')

# 输出文件路径
VALID_TAGS_PATH = os.path.join(DATA_DIR, 'valid_tags.json')
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_data.csv')
VAL_DATA_PATH = os.path.join(DATA_DIR, 'val_data.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_data.csv')
PROCESSED_FULL_DATA_PATH = os.path.join(DATA_DIR, 'processed_full_data.csv')

# 特征文件路径
TFIDF_VECTORIZER_PATH = os.path.join(DATA_DIR, 'tfidf_vectorizer.pkl')
X_TRAIN_TFIDF_PATH = os.path.join(DATA_DIR, 'X_train_tfidf.npz')
X_VAL_TFIDF_PATH = os.path.join(DATA_DIR, 'X_val_tfidf.npz')
X_TEST_TFIDF_PATH = os.path.join(DATA_DIR, 'X_test_tfidf.npz')

# 模型文件路径
BILSTM_MODEL_PATH = os.path.join(OUTPUT_DIR, 'bilstm_model.pt')
BILSTM_VOCAB_PATH = os.path.join(OUTPUT_DIR, 'bilstm_vocab.json')
BILSTM_RESULTS_PATH = os.path.join(OUTPUT_DIR, 'bilstm_results.json')

# 图表文件路径
TEXT_LENGTH_DIST_PATH = os.path.join(OUTPUT_DIR, 'text_length_dist.png')
TOP_TAGS_DIST_PATH = os.path.join(DATA_DIR, 'top_tags_dist.png')
SYNOPSIS_LENGTH_DIST_PATH = os.path.join(DATA_DIR, 'synopsis_length_dist.png')
BILSTM_TOP_TAGS_PERFORMANCE_PATH = os.path.join(OUTPUT_DIR, 'bilstm_top_tags_performance.png')
BILSTM_LOSS_CURVE_PATH = os.path.join(OUTPUT_DIR, 'bilstm_loss_curve.png')

# 模型训练设置
SEED = 42                # 随机种子
MAX_VOCAB_SIZE = 30000   # 最大词汇表大小
MAX_SEQ_LENGTH = 300     # 最大序列长度
BATCH_SIZE = 32          # 批量大小
EMBEDDING_DIM = 300      # 词嵌入维度
HIDDEN_DIM = 128         # 隐藏层维度
N_LAYERS = 2             # LSTM层数
DROPOUT = 0.5            # Dropout比率
LEARNING_RATE = 0.001    # 学习率
WEIGHT_DECAY = 1e-5      # L2正则化系数
MAX_EPOCHS = 20          # 最大训练轮次（所有模型通用的训练轮次参数）
PATIENCE = 5             # 早停耐心值
PREDICTION_THRESHOLD = 0.2  # 多标签分类预测阈值，所有模型共用

# ============= BERT 模型配置 =============
# 根据操作系统设置路径格式
if platform.system() == 'Windows':
    # Windows系统使用这种路径格式
    LOCAL_BERT_MODEL_PATH = 'D:/share/base/bert-base-uncased'  # BERT模型本地路径
    BERT_3TAGS_DIR = 'D:/share/base/bert-3tags'               # BERT输出目录
else:
    # Mac/Linux系统使用这种路径格式
    LOCAL_BERT_MODEL_PATH = '/Volumes/share/base/bert-base-uncased'
    BERT_3TAGS_DIR = '/Volumes/share/base/bert-3tags'

# 确保BERT输出目录存在
os.makedirs(BERT_3TAGS_DIR, exist_ok=True)

# BERT模型设置
BERT_MODEL_NAME = 'bert-base-uncased'  # 使用的BERT模型名称
BERT_MAX_LENGTH = 128                  # BERT输入的最大长度
BERT_LEARNING_RATE = 2e-5              # BERT层学习率
CLASSIFIER_LEARNING_RATE = 1e-3        # 分类器学习率
WARMUP_PROPORTION = 0.1                # 预热比例
BERT_DROPOUT = 0.3                     # BERT模型的dropout率
CPU_BATCH_SIZE = 8                     # CPU训练时的批处理大小
OFFLINE_MODE = False                   # 如果为True，将不会从Hugging Face下载模型

# BERT输出文件路径
BERT_BEST_MODEL_PATH = os.path.join(BERT_3TAGS_DIR, 'best_model.pt')
BERT_WORST_MODEL_PATH = os.path.join(BERT_3TAGS_DIR, 'worst_model.pt')
BERT_3TAGS_RESULTS_PATH = os.path.join(BERT_3TAGS_DIR, 'results.json')
BERT_3TAGS_LOSS_CURVE_PATH = os.path.join(BERT_3TAGS_DIR, 'loss_curves.png')
BERT_3TAGS_CONFUSION_MATRIX_PATH = os.path.join(BERT_3TAGS_DIR, 'confusion_matrix.png')
BERT_3TAGS_TAG_CORR_PATH = os.path.join(BERT_3TAGS_DIR, 'tag_correlations.png')
BERT_3TAGS_PERFORMANCE_DIST_PATH = os.path.join(BERT_3TAGS_DIR, 'performance_distribution.png')
BERT_3TAGS_TOP_TAGS_PATH = os.path.join(BERT_3TAGS_DIR, 'top_tags_performance.png')
BERT_3TAGS_EPOCH_METRICS_PATH = os.path.join(BERT_3TAGS_DIR, 'epoch_metrics.png')
BERT_3TAGS_TAG_HEATMAP_PATH = os.path.join(BERT_3TAGS_DIR, 'tag_heatmap.png')
BERT_TEXT_LENGTH_DIST_PATH = os.path.join(BERT_3TAGS_DIR, 'text_length_dist.png')
BERT_TOP_TAGS_DIST_PATH = os.path.join(BERT_3TAGS_DIR, 'top_tags_distribution.png')
BERT_SAMPLE_PRECISION_DIST_PATH = os.path.join(BERT_3TAGS_DIR, 'sample_precision_dist.png')
BERT_TAG_FREQ_COMPARISON_PATH = os.path.join(BERT_3TAGS_DIR, 'tag_freq_comparison.png')
BERT_TRAINING_METRICS_PATH = os.path.join(BERT_3TAGS_DIR, 'training_metrics.json')
BERT_MODEL_CONFIG_PATH = os.path.join(BERT_3TAGS_DIR, 'model_config.json') 
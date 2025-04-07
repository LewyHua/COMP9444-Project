#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MPST数据集处理与模型训练配置文件
此文件包含所有路径设置，可根据实际需求进行修改
"""

import os

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
MAX_EPOCHS = 20          # 最大训练轮次
PATIENCE = 5             # 早停耐心值 
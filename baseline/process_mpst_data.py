#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MPST_v2 电影简介分类数据集预处理脚本
包含：数据分析、文本清理、标签处理和数据集划分
"""

import os
import re
import json
import html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import pickle
import time

# 导入配置文件中的路径设置
from config import (
    INPUT_DIR, DATA_DIR, 
    TRAIN_IDS_PATH, TEST_IDS_PATH, ALL_DATA_PATH,
    VALID_TAGS_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH,
    PROCESSED_FULL_DATA_PATH, TFIDF_VECTORIZER_PATH,
    X_TRAIN_TFIDF_PATH, X_VAL_TFIDF_PATH, X_TEST_TFIDF_PATH,
    TOP_TAGS_DIST_PATH, SYNOPSIS_LENGTH_DIST_PATH
)

# 确保输出目录存在
os.makedirs(DATA_DIR, exist_ok=True)

# 下载必要的NLTK资源
def download_nltk_resources():
    """下载并确保NLTK资源可用"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

# 1. 数据加载与分析
def load_and_analyze_data():
    """加载MPST_v2数据集并进行基本分析"""
    print("正在加载MPST_v2数据集...")
    
    # 加载主数据文件
    df = pd.read_csv(ALL_DATA_PATH)
    print(f"数据加载完成，形状: {df.shape}")
    
    # 显示基本信息
    print(f"\n数据集基本信息:")
    print(f"列名: {df.columns.tolist()}")
    
    # 检查缺失值
    missing_values = df.isnull().sum()
    print("\n缺失值统计:")
    print(missing_values)
    
    # 统计基本信息
    print("\n数据集统计信息:")
    for col in df.columns:
        if df[col].dtype == 'object':
            # 对于文本列，计算非空值数量和平均长度
            non_null = df[col].notnull().sum()
            if non_null > 0:
                avg_len = df[col].astype(str).apply(len).mean()
                print(f"{col}: {non_null} 非空值, 平均长度 {avg_len:.2f}")
        else:
            # 对于数值列，显示基本统计信息
            print(f"{col}: {df[col].describe()}")
    
    # 分析文本长度分布
    if 'plot_synopsis' in df.columns:
        df['synopsis_length'] = df['plot_synopsis'].astype(str).apply(len)
        plt.figure(figsize=(10, 6))
        sns.histplot(df['synopsis_length'], bins=50)
        plt.title('Synopsis Length Distribution')
        plt.xlabel('Length (characters)')
        plt.ylabel('Count')
        plt.savefig(SYNOPSIS_LENGTH_DIST_PATH)
        print(f"\n简介长度统计信息:\n{df['synopsis_length'].describe()}")
    
    # 分析标签分布
    if 'tags' in df.columns:
        # 假设标签以某种分隔符存储，这里假设是逗号
        df['num_tags'] = df['tags'].astype(str).apply(lambda x: len(x.split(',')))
        print(f"\n每部电影的平均标签数量: {df['num_tags'].mean():.2f}")
        
        # 统计标签频率
        all_tags = []
        for tags in df['tags'].astype(str):
            all_tags.extend([tag.strip() for tag in tags.split(',')])
        
        tag_counts = Counter(all_tags)
        print(f"\n前20个最常见的标签:")
        for tag, count in tag_counts.most_common(20):
            print(f"{tag}: {count}")
        
        # 绘制标签分布图
        plt.figure(figsize=(12, 8))
        top_tags = dict(tag_counts.most_common(20))
        plt.bar(top_tags.keys(), top_tags.values())
        plt.title('Top 20 Tags Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(TOP_TAGS_DIST_PATH)
    
    return df

# 2. 文本清理
def clean_text(text):
    """清理文本，去除HTML标签、特殊字符等"""
    if not isinstance(text, str):
        return ""
    
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 处理HTML实体
    text = html.unescape(text)
    
    # 替换URL
    text = re.sub(r'http\S+', '[URL]', text)
    
    # 替换非标准字符和符号
    text = re.sub(r'[^\w\s.,!?;:()\-\'\"]', ' ', text)
    
    # 规范化空白
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text(df, text_column='plot_synopsis'):
    """对数据集中的文本列进行预处理"""
    print(f"\n正在预处理文本列 '{text_column}'...")
    
    if text_column not in df.columns:
        print(f"错误: 列 '{text_column}' 不存在")
        return df
    
    # 确保NLTK资源已下载
    download_nltk_resources()
    
    # 应用基本清理
    df['cleaned_text'] = df[text_column].apply(clean_text)
    
    # 转换为小写
    df['cleaned_text'] = df['cleaned_text'].str.lower()
    
    # 获取英文停用词
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # 应用分词、停用词移除和词形还原
    def process_text(text):
        if not isinstance(text, str) or not text:
            return [], ""
        
        # 分词
        tokens = word_tokenize(text)
        
        # 移除停用词并执行词形还原
        filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        
        return filtered_tokens, ' '.join(filtered_tokens)
    
    # 应用处理函数
    result = df['cleaned_text'].apply(process_text)
    df['tokens'] = result.apply(lambda x: x[0])
    df['processed_text'] = result.apply(lambda x: x[1])
    
    # 统计处理前后的文本长度
    df['original_length'] = df[text_column].astype(str).apply(len)
    df['processed_length'] = df['processed_text'].astype(str).apply(len)
    length_ratio = (df['processed_length'] / df['original_length']).mean()
    print(f"处理后的文本平均长度为原始文本的 {length_ratio:.2%}")
    
    # 移除空文本
    empty_texts = (df['processed_text'] == '').sum()
    print(f"空文本数量: {empty_texts}")
    
    return df

# 3. 标签处理
def process_tags(df, tag_column='tags'):
    """处理标签，包括标准化和编码"""
    print(f"\n正在处理标签列 '{tag_column}'...")
    
    if tag_column not in df.columns:
        print(f"错误: 列 '{tag_column}' 不存在")
        return df
    
    # 创建标签映射字典 - 这里应根据实际数据调整
    # 这只是一个示例，实际的标签映射应该根据数据分析决定
    tag_mapping = {
        'sci-fi': 'science fiction',
        'rom com': 'romantic comedy',
        'romcom': 'romantic comedy',
        'scifi': 'science fiction',
        'drama/comedy': 'comedy,drama',
        # 添加更多映射...
    }
    
    # 处理标签：分割、清理、标准化
    def standardize_tags(tag_str):
        if not isinstance(tag_str, str) or not tag_str:
            return []
        
        # 分割标签
        tags = [t.strip().lower() for t in tag_str.split(',')]
        
        # 应用映射
        mapped_tags = [tag_mapping.get(t, t) for t in tags]
        
        # 再次分割可能的组合标签
        final_tags = []
        for tag in mapped_tags:
            if ',' in tag:
                final_tags.extend([t.strip() for t in tag.split(',')])
            else:
                final_tags.append(tag)
        
        return sorted(set(final_tags))  # 移除重复
    
    # 应用标准化
    df['standardized_tags'] = df[tag_column].apply(standardize_tags)
    
    # 统计标准化后的标签
    all_tags = []
    for tags in df['standardized_tags']:
        all_tags.extend(tags)
    
    tag_counts = Counter(all_tags)
    print(f"\n标准化后共有 {len(tag_counts)} 个不同的标签")
    print(f"前20个最常见的标准化标签:")
    for tag, count in tag_counts.most_common(20):
        print(f"{tag}: {count}")
    
    # 选择出现频率超过阈值的标签
    min_count = 50  # 可调整的阈值
    valid_tags = [tag for tag, count in tag_counts.items() if count >= min_count]
    print(f"\n选择出现频率 >= {min_count} 的标签: {len(valid_tags)} 个")
    
    # 创建多标签二值表示
    for tag in valid_tags:
        col_name = f'tag_{tag.replace(" ", "_")}'
        df[col_name] = df['standardized_tags'].apply(lambda x: 1 if tag in x else 0)
    
    return df, valid_tags

# 4. 数据集划分
def split_dataset(df, train_ids_path=TRAIN_IDS_PATH, test_ids_path=TEST_IDS_PATH, id_column='imdb_id'):
    """根据提供的ID列表划分数据集"""
    print("\n正在划分数据集...")
    
    # 检查ID列是否存在
    if id_column not in df.columns:
        print(f"错误: ID列 '{id_column}' 不存在")
        return df, None, None, None
    
    # 加载训练集和测试集ID
    try:
        with open(train_ids_path, 'r') as f:
            train_ids = set(line.strip() for line in f)
        with open(test_ids_path, 'r') as f:
            test_ids = set(line.strip() for line in f)
        
        print(f"加载了 {len(train_ids)} 个训练ID和 {len(test_ids)} 个测试ID")
    except FileNotFoundError:
        print(f"警告: ID文件不存在，使用随机划分")
        # 如果找不到ID文件，使用随机划分
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
        return df, train_df, val_df, test_df
    
    # 根据ID划分数据集
    train_df = df[df[id_column].isin(train_ids)]
    test_all_df = df[df[id_column].isin(test_ids)]
    
    # 将测试集进一步划分为验证集和测试集
    val_df, test_df = train_test_split(test_all_df, test_size=0.55, random_state=42)
    
    print(f"数据集划分完成:")
    print(f"训练集: {len(train_df)} 样本 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"验证集: {len(val_df)} 样本 ({len(val_df)/len(df)*100:.1f}%)")
    print(f"测试集: {len(test_df)} 样本 ({len(test_df)/len(df)*100:.1f}%)")
    
    return df, train_df, val_df, test_df

# 5. 特征提取
def extract_features(train_df, val_df, test_df, text_column='processed_text'):
    """提取文本特征"""
    print("\n正在提取TF-IDF特征...")
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # 创建TF-IDF向量器
    tfidf = TfidfVectorizer(
        max_features=5000,      # 最大特征数
        min_df=5,               # 忽略在少于5个文档中出现的词
        max_df=0.95,            # 忽略在超过95%文档中出现的词
        stop_words='english',   # 移除英文停用词
        ngram_range=(1, 2),     # 使用1-gram和2-gram
        use_idf=True,           # 使用逆文档频率
        sublinear_tf=True       # 对词频使用次线性缩放
    )
    
    # 在训练集上拟合向量器
    X_train = tfidf.fit_transform(train_df[text_column].fillna(''))
    
    # 转换验证集和测试集
    X_val = tfidf.transform(val_df[text_column].fillna(''))
    X_test = tfidf.transform(test_df[text_column].fillna(''))
    
    print(f"TF-IDF特征提取完成:")
    print(f"训练集: {X_train.shape}")
    print(f"验证集: {X_val.shape}")
    print(f"测试集: {X_test.shape}")
    
    # 保存向量器和特征
    with open(TFIDF_VECTORIZER_PATH, 'wb') as f:
        pickle.dump(tfidf, f)
    
    # 保存稀疏矩阵
    from scipy import sparse
    sparse.save_npz(X_TRAIN_TFIDF_PATH, X_train)
    sparse.save_npz(X_VAL_TFIDF_PATH, X_val)
    sparse.save_npz(X_TEST_TFIDF_PATH, X_test)
    
    # 返回特征信息
    return {
        'vectorizer': tfidf,
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test
    }

# 6. 保存处理后的数据
def save_processed_data(df, train_df, val_df, test_df, valid_tags):
    """保存处理后的数据到输出目录"""
    print("\n正在保存处理后的数据...")
    
    # 保存完整的处理后数据集
    df.to_csv(PROCESSED_FULL_DATA_PATH, index=False)
    
    # 保存划分后的数据集
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    val_df.to_csv(VAL_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    
    # 保存标签信息
    with open(VALID_TAGS_PATH, 'w') as f:
        json.dump(valid_tags, f)
    
    # 保存处理配置和统计信息
    stats = {
        'total_samples': len(df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'num_tags': len(valid_tags),
        'top_tags': dict(Counter([tag for tags in df['standardized_tags'] for tag in tags]).most_common(20)),
        'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(DATA_DIR, 'processing_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("数据保存完成")
    return stats

# 7. 主函数
def main():
    """主处理流程"""
    start_time = time.time()
    
    print("="*80)
    print("开始MPST_v2电影简介分类数据集预处理")
    print("="*80)
    
    # 1. 加载和分析数据
    df = load_and_analyze_data()
    
    # 2. 文本预处理
    df = preprocess_text(df, text_column='plot_synopsis')
    
    # 3. 标签处理
    df, valid_tags = process_tags(df, tag_column='tags')
    
    # 4. 数据集划分
    df, train_df, val_df, test_df = split_dataset(df)
    
    # 5. 特征提取
    feature_info = extract_features(train_df, val_df, test_df)
    
    # 6. 保存处理后的数据
    stats = save_processed_data(df, train_df, val_df, test_df, valid_tags)
    
    # 计算总处理时间
    elapsed_time = time.time() - start_time
    print(f"\n数据预处理完成，总耗时: {elapsed_time:.2f} 秒")
    print(f"处理后的数据已保存到: {DATA_DIR}")
    
    return df, feature_info, stats

if __name__ == "__main__":
    main() 
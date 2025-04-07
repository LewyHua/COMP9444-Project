# MPST电影标签分类项目

基于"MPST: A Corpus of Movie Plot Synopses with Tags"论文，实现电影剧情简介的多标签分类。

## 项目结构

```
.
└── baseline/              # 基准模型实现
    ├── MPST_v2/           # 原始数据集目录
    ├── dataset/           # 处理后的数据目录
    │   ├── processed_full_data.csv      # 完整处理后数据
    │   ├── train_data.csv               # 训练集
    │   ├── val_data.csv                 # 验证集
    │   ├── test_data.csv                # 测试集
    │   ├── valid_tags.json              # 有效标签列表
    │   └── processing_stats.json        # 处理统计信息
    ├── models/            # 模型输出目录
    │   ├── bilstm_model.pt              # BiLSTM模型参数
    │   ├── bilstm_vocab.json            # 词汇表
    │   ├── bilstm_results.json          # 模型评估结果
    │   └── bilstm_top_tags_performance.png  # 性能图表
    ├── process_mpst_data.py   # 数据预处理脚本
    ├── train_bilstm_classifier.py  # BiLSTM模型训练脚本
    ├── download_data.py       # 数据集下载脚本
    ├── config.py              # 配置文件（路径设置）
    ├── README.md              # 项目说明文档
    └── requirements.txt       # 依赖项
```

## 安装依赖

```bash
cd baseline
pip install -r requirements.txt
```

## 下载数据集和模型

由于GitHub对大文件（>100MB）有限制，本项目的数据文件和模型文件存储在Google Drive中。您可以使用提供的下载脚本轻松获取这些文件：

```bash
cd baseline
python download_data.py
```

此脚本将：
1. 检查并安装必要的依赖库（gdown）
2. 从Google Drive下载所有数据文件和模型文件
3. 自动将文件放置在正确的目录结构中

**注意**：下载速度取决于您的网络连接和Google Drive的响应速度。大型文件（如训练数据集）可能需要较长时间下载。

如果脚本无法正常工作，您也可以手动从以下链接下载：
- [原始MPST数据集](https://drive.google.com/file/d/17Y4XQu1pEYEdiMWs1ARnZGqE_soWK0Tu/view?usp=sharing) - 用于从头开始处理
- [预处理数据文件](https://drive.google.com/drive/folders/1S3O_VXz_55DjdpNofk5YPpSsVqbv7DlF?usp=sharing) - 包含预处理数据、TF-IDF特征等
- [BiLSTM模型文件](https://drive.google.com/drive/folders/1hEqnI49j0IodiwgToOfotToJ8BISGY8E?usp=sharing) - 包含预训练的模型权重

手动下载后，请将文件放置在正确的目录中：
- 原始数据集文件解压后 → `./baseline/MPST_v2/`目录
- 预处理数据文件（`*.csv`, `*.json`, `*.npz`等）→ `./baseline/dataset/`目录
- 模型文件（`bilstm_model.pt`）→ `./baseline/models/`目录

## 数据处理流程

您可以选择使用预处理好的数据文件，或者从原始MPST数据集开始处理：

### 从原始数据集开始

1. 下载原始MPST数据集并解压到`./baseline/MPST_v2/`目录
2. 运行预处理脚本：
   ```bash
   cd baseline
   python process_mpst_data.py
   ```
3. 预处理后的数据将保存到`./baseline/dataset/`目录

### 使用预处理好的数据

1. 下载预处理数据文件到`./baseline/dataset/`目录
2. 直接进行模型训练或使用预训练模型

## 配置文件使用说明

项目使用`config.py`文件集中管理所有路径设置和模型参数。您可以根据自己的需求修改此文件中的路径变量：

### 主要配置项

- **数据目录设置**：
  - `INPUT_DIR`：原始MPST数据集目录（默认: './MPST_v2'）
  - `DATA_DIR`：处理后数据输出目录（默认: './dataset'）
  - `OUTPUT_DIR`：模型和结果输出目录（默认: './models'）

- **模型训练设置**：
  - `MAX_VOCAB_SIZE`：最大词汇表大小（默认: 30000）
  - `MAX_SEQ_LENGTH`：最大序列长度（默认: 300）
  - `BATCH_SIZE`：批量大小（默认: 32）
  - `MAX_EPOCHS`：最大训练轮次（默认: 50）

### 使用方法

1. 修改配置（可选）：
   ```python
   # 在config.py中修改配置
   INPUT_DIR = '/path/to/your/mpst_data'  # 修改为您的数据目录
   OUTPUT_DIR = '/path/to/save/models'    # 修改为您的模型保存目录
   MAX_EPOCHS = 100                       # 增加训练轮次
   ```

2. 在脚本中导入配置：
   ```python
   # 在您的脚本中
   from config import DATA_DIR, OUTPUT_DIR, MAX_EPOCHS
   
   # 使用配置变量
   print(f"数据目录: {DATA_DIR}")
   print(f"将在 {MAX_EPOCHS} 轮次内训练模型")
   ```

## 使用预训练模型

如果您不想自己训练模型，可以直接使用我们提供的预训练BiLSTM模型：

1. 下载模型文件（通过脚本或手动从链接）
2. 确保模型文件`bilstm_model.pt`位于`./baseline/models/`目录中
3. 加载并使用模型进行预测：

```python
import torch
from train_bilstm_classifier import BiLSTMWithAttention
from config import BILSTM_MODEL_PATH, BILSTM_VOCAB_PATH

# 加载词汇表
with open(BILSTM_VOCAB_PATH, 'r') as f:
    word_to_idx = json.load(f)

# 初始化模型
model = BiLSTMWithAttention(
    vocab_size=len(word_to_idx),
    embedding_dim=300,
    hidden_dim=128,
    output_dim=68,
    n_layers=2,
    dropout=0.5
)

# 加载预训练权重
model.load_state_dict(torch.load(BILSTM_MODEL_PATH, map_location=torch.device('cpu')))
model.eval()  # 设置为评估模式

# 现在可以使用模型进行预测
```

## 数据处理

执行以下命令处理MPST数据集：

```bash
cd baseline
python process_mpst_data.py
```

此脚本将：
1. 加载MPST_v2数据集
2. 清洗和预处理文本数据
3. 处理和标准化标签
4. 将数据集分割为训练/验证/测试集
5. 提取TF-IDF特征
6. 将处理后的数据保存到`DATA_DIR`目录

## 模型训练

执行以下命令训练BiLSTM模型：

```bash
cd baseline
python train_bilstm_classifier.py
```

此脚本将：
1. 加载处理好的数据
2. 构建词汇表和数据加载器
3. 初始化BiLSTM+注意力模型
4. 训练模型（最多`MAX_EPOCHS`轮）
5. 评估模型在验证集和测试集上的性能
6. 将模型和结果保存到`OUTPUT_DIR`目录

## 结果说明

训练后，可以在`baseline/models/`目录中找到：
- 模型参数文件：`bilstm_model.pt`
- 评估结果：`bilstm_results.json`
- 性能可视化：`bilstm_top_tags_performance.png`和`bilstm_loss_curve.png`

## 注意事项

- 首次运行时会下载NLTK资源，需要网络连接
- 训练BiLSTM模型可能需要较长时间，请耐心等待
- 如果您的系统有GPU，将自动使用GPU进行训练
- 预训练模型的大小约为69MB，请确保有足够的存储空间

## 数据集引用

如果您使用此项目，请引用原始MPST数据集：

```
@inproceedings{Kar2018,
  title={MPST: A Corpus of Movie Plot Synopses with Tags},
  author={Kar, Sudipta and Maharjan, Suraj and Solorio, Thamar},
  booktitle={Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
  year={2018}
}
```
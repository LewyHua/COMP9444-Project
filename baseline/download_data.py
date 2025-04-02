#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从Google Drive下载数据集文件的脚本
使用gdown库从Google Drive下载整个文件夹和文件
"""

import os
import sys
import shutil
import subprocess
from tqdm import tqdm
import py7zr  # 替换zipfile，用于处理.7z文件

# 确保输出目录存在
from config import INPUT_DIR, DATA_DIR, OUTPUT_DIR
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Google Drive文件夹和文件ID
DATASET_FOLDER_ID = "1S3O_VXz_55DjdpNofk5YPpSsVqbv7DlF"  # 预处理数据文件夹
MODEL_FOLDER_ID = "1hEqnI49j0IodiwgToOfotToJ8BISGY8E"     # 模型文件夹
ORIGINAL_DATASET_FILE_ID = "17Y4XQu1pEYEdiMWs1ARnZGqE_soWK0Tu"  # 原始MPST数据集文件

def install_gdown_if_needed():
    """检查并安装gdown"""
    try:
        import gdown
        # 检查gdown版本，确保支持文件夹下载
        version = gdown.__version__
        print(f"已安装gdown版本: {version}")
        # 如果需要升级gdown
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "gdown"], check=True)
        return True
    except ImportError:
        print("gdown库未安装，正在安装...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
            return True
        except Exception as e:
            print(f"安装gdown时出错: {e}")
            return False

def install_py7zr_if_needed():
    """检查并安装py7zr"""
    try:
        import py7zr
        version = py7zr.__version__
        print(f"已安装py7zr版本: {version}")
        return True
    except ImportError:
        print("py7zr库未安装，正在安装...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "py7zr"], check=True)
            return True
        except Exception as e:
            print(f"安装py7zr时出错: {e}")
            return False

def download_folder(folder_id, output_dir):
    """下载整个Google Drive文件夹"""
    # 导入gdown
    import gdown
    
    print(f"正在从Google Drive下载文件夹...")
    try:
        # 创建临时目录
        temp_dir = "temp_download"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 使用gdown下载文件夹
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        gdown.download_folder(url=url, output=temp_dir, quiet=False, use_cookies=False)
        
        print("下载完成！正在整理文件...")
        
        # 将数据文件移动到正确的位置
        dataset_files = [f for f in os.listdir(temp_dir) if not f.startswith('.')]
        
        for file in dataset_files:
            src_path = os.path.join(temp_dir, file)
            
            # 确定目标路径
            if file == "bilstm_model.pt":
                # 模型文件放在models目录
                dst_path = os.path.join(OUTPUT_DIR, file)
            else:
                # 其他文件放在dataset目录
                dst_path = os.path.join(DATA_DIR, file)
            
            # 如果目标文件已存在，先删除
            if os.path.exists(dst_path):
                if os.path.isfile(dst_path):
                    os.remove(dst_path)
                else:
                    shutil.rmtree(dst_path)
            
            # 移动文件
            shutil.move(src_path, dst_path)
            print(f"已移动: {file} -> {dst_path}")
        
        # 清理临时目录
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"下载文件夹时出错: {e}")
        return False

def download_file(file_id, output_path, desc):
    """从Google Drive下载单个文件"""
    import gdown
    
    print(f"正在下载{desc}...")
    url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        print(f"下载{desc}时出错: {e}")
        return False

def extract_7z(archive_path, extract_to):
    """解压7z文件"""
    print(f"正在解压文件到 {extract_to}...")
    try:
        # 打开7z文件
        with py7zr.SevenZipFile(archive_path, mode='r') as archive:
            # 获取文件列表用于显示进度
            file_list = archive.getnames()
            total_files = len(file_list)
            
            # 使用tqdm显示进度
            with tqdm(total=total_files) as pbar:
                # 解压所有文件
                archive.extractall(path=extract_to)
                # 由于py7zr不支持单文件提取的回调，我们一次性更新进度条
                pbar.update(total_files)
        
        print("解压完成！")
        return True
    except Exception as e:
        print(f"解压文件时出错: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("MPST数据集文件下载工具")
    print("=" * 60)
    
    # 安装或升级gdown
    if not install_gdown_if_needed():
        print("安装gdown失败，无法继续。")
        return
    
    # 安装py7zr
    if not install_py7zr_if_needed():
        print("安装py7zr失败，无法继续。")
        return
    
    # 下载选项菜单
    print("\n请选择要下载的内容:")
    print("1. 所有文件（原始数据集、预处理数据和模型）")
    print("2. 仅原始MPST数据集")
    print("3. 仅预处理好的数据文件")
    print("4. 仅预训练模型文件")
    
    try:
        choice = int(input("\n请输入选项 [1-4] (默认: 1): ") or "1")
    except ValueError:
        choice = 1
        print("输入无效，使用默认选项1")
    
    # 下载原始MPST数据集
    if choice in [1, 2]:
        print("\n==== 下载原始MPST数据集 ====")
        original_dataset_path = "MPST_v2.7z"  # 改为.7z扩展名
        success = download_file(ORIGINAL_DATASET_FILE_ID, original_dataset_path, "原始MPST数据集")
        if success:
            # 解压到INPUT_DIR
            if extract_7z(original_dataset_path, "."):
                print(f"原始数据集已解压到 {INPUT_DIR} 目录")
                # 删除7z文件
                os.remove(original_dataset_path)
    
    # 下载预处理数据文件
    if choice in [1, 3]:
        print("\n==== 下载预处理数据文件 ====")
        success = download_folder(DATASET_FOLDER_ID, ".")
        if not success:
            print("下载预处理数据文件失败，请稍后重试或手动下载")
    
    # 下载模型文件
    if choice in [1, 4]:
        print("\n==== 下载模型文件 ====")
        success = download_folder(MODEL_FOLDER_ID, ".")
        if not success:
            print("下载模型文件失败，请稍后重试或手动下载")
    
    print("\n所有下载操作完成！")
    print("如果下载失败，您可以参考README中的链接手动下载文件。")

if __name__ == "__main__":
    main() 
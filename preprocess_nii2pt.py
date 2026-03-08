import os
import glob
import pandas as pd
import numpy as np
import nibabel as nib
import torch
import scipy.ndimage
import re  # 引入正则用于提取Subject ID
from tqdm import tqdm

# ================= 配置区域 =================
# 1. .nii.gz 文件所在的文件夹
RAW_DATA_DIR = r"/root/autodl-tmp/hdbet_output/" 

# 2. labels.csv 文件路径
CSV_PATH = r"/root/autodl-tmp/labels.csv"

# 3. 输出路径
OUTPUT_DIR = r"/root/autodl-tmp/pt_output/"

# 4. 目标尺寸
TARGET_SHAPE = (128, 128, 128)

# 5. CN 和 AD严格匹配
GROUP_MAP = {
    'CN': 'CN',
    'AD': 'AD',
}
# ===========================================

def load_and_preprocess(path):
    """
    读取 nifti -> 归一化 -> 裁剪/缩放 -> 转 Tensor
    """
    try:
        img_obj = nib.load(path)
        img_data = img_obj.get_fdata().astype(np.float32)

        # 1. 裁剪背景 (Crop)
        non_zero = np.where(img_data > 0)
        if len(non_zero[0]) == 0: return None
        
        min_x, max_x = np.min(non_zero[0]), np.max(non_zero[0])
        min_y, max_y = np.min(non_zero[1]), np.max(non_zero[1])
        min_z, max_z = np.min(non_zero[2]), np.max(non_zero[2])
        
        img_crop = img_data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]

        # 2. 缩放 (Resize) 到 128x128x128
        current_shape = img_crop.shape
        zoom_factors = [t / c for t, c in zip(TARGET_SHAPE, current_shape)]
        img_resized = scipy.ndimage.zoom(img_crop, zoom_factors, order=1)

        # 3. 强度标准化 (Z-score)
        mean = np.mean(img_resized)
        std = np.std(img_resized)
        if std > 0:
            img_norm = (img_resized - mean) / std
        else:
            img_norm = img_resized

        # 4. 转 Tensor [1, 128, 128, 128]
        img_tensor = torch.from_numpy(img_norm).float().unsqueeze(0)
        return img_tensor

    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 读取 CSV 并构建联合查找字典
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    
    # 确保列名存在（请根据你实际csv列名调整，比如 'Subject' 或 'Subject ID'）
    if 'Image Data ID' not in df.columns or 'Group' not in df.columns or 'Subject' not in df.columns:
        print("Error: CSV 必须包含 'Subject', 'Image Data ID', 'Group' 列")
        return

    # 构建字典：Key = (Subject字符串, ImageID字符串), Value = Group
    # 这样可以同时通过 Subject 和 ImageID 唯一锁定一行
    label_dict = {}
    for idx, row in df.iterrows():
        sub = str(row['Subject']).strip()
        # CSV里的ID是数字，转成字符串 (例如 12345 -> "12345")
        img_id = str(row['Image Data ID']).strip() 
        group = row['Group']
        
        # 存入字典
        label_dict[(sub, img_id)] = group

    print(f"Loaded {len(label_dict)} entries from CSV.")

    # 2. 遍历文件
    nii_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.nii.gz"))
    print(f"Found {len(nii_files)} .nii.gz files.")

    success_count = 0
    skip_count = 0
    
    for nii_path in tqdm(nii_files):
        filename = os.path.basename(nii_path)
        
        # --- 解析文件名信息 ---
        # 1. 提取 Image ID (匹配 I后面跟一串数字)
        # group(1) 会提取出 '17899' (不包含I，也不包含后缀)
        match_id = re.search(r'I(\d+)', filename)
        file_img_id = match_id.group(1) if match_id else None
        
        # 2. 提取 Subject ID (002_S_xxxx)
        # 使用正则提取标准 ADNI 格式: 3位数字_S_4位数字
        match_subject = re.search(r'(\d{3}_S_\d{4})', filename)
        file_subject_id = match_subject.group(1) if match_subject else None

        # --- 匹配逻辑 ---
        if not file_img_id or not file_subject_id:
            # print(f"Skipping {filename}: Cannot parse Subject or ImageID")
            skip_count += 1
            continue

        # 【修改点】使用 (Subject, ImageID) 双重查找
        key = (file_subject_id, file_img_id)
        
        if key not in label_dict:
            # print(f"Skipping {filename}: Key {key} not found in CSV")
            skip_count += 1
            continue

        raw_group = label_dict[key]

        # --- 筛选 Group ---
        if raw_group not in GROUP_MAP:
            # 如果是 MCI 或其他不在 GROUP_MAP 里的类别，直接跳过
            skip_count += 1
            continue
            
        final_group = GROUP_MAP[raw_group]

        # --- 预处理 ---
        tensor = load_and_preprocess(nii_path)
        
        if tensor is not None:
            save_name = f"{file_subject_id}_{file_img_id}_{final_group}.pt"
            save_path = os.path.join(OUTPUT_DIR, save_name)
          
            torch.save(tensor, save_path)
            success_count += 1
        else:
            print(f"Failed to preprocess: {filename}")

    print(f"\nProcessing Complete!")
    print(f"Converted: {success_count}")
    print(f"Skipped: {skip_count} (Not in CSV, unknown group, or parsing error)")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
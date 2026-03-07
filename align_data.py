import os
import pandas as pd
import re

# ================= 1. 路径配置区 =================
IMG_DIR = '/root/autodl-tmp/hdbet_output/'   # 你的图像文件夹路径
CSV_INPUT = '/root/autodl-tmp/labels_raw.csv' # 原始的 CSV 标签文件
CSV_OUTPUT = '/root/autodl-tmp/labels.csv'   # 筛选后生成的新 CSV 文件
# ===============================================

def main():
    print("🚀 开始进行数据双向对齐检查...\n")

    # ================= 步骤 A：解析文件夹中的真实文件 =================
    # 获取目录下所有 hdbet_ 开头，且不是 mask 掩码的文件
    all_files =[f for f in os.listdir(IMG_DIR) if f.startswith('hdbet_') and not f.endswith('mask.nii.gz')]
    
    valid_files_set = set()
    # 匹配规则：hdbet_受试者ID_I图像ID.nii.gz
    # 正则提取 Subject (如 002_S_0339) 和 Image ID (提取 I 后面的纯数字，如 113375)
    file_pattern = re.compile(r'hdbet_(.*)_I(\d+)\.nii')
    
    for f in all_files:
        match = file_pattern.search(f)
        if match:
            subject_id = match.group(1).strip()
            image_id = match.group(2).strip() # 这里提取出来的纯数字，刚好匹配 CSV 里的格式
            valid_files_set.add((subject_id, image_id))
            
    print(f"📁 步骤A：在文件夹中扫描到 {len(valid_files_set)} 个有效的脑部 MRI 图像。")

    # ================= 步骤 B：处理 CSV 并剔除多余的行 =================
    df = pd.read_csv(CSV_INPUT)
    
    # 极其重要：清洗 CSV 中的空格，并强制转换为字符串类型！
    # 防止 pandas 把 113375 当作整型，或者包含不可见的空格导致匹配失败
    df['Subject'] = df['Subject'].astype(str).str.strip()
    df['Image Data ID'] = df['Image Data ID'].astype(str).str.strip()
    
    # 筛选条件：这一行的 (Subject, Image Data ID) 必须在文件夹的集合中存在
    def is_in_folder(row):
        return (row['Subject'], row['Image Data ID']) in valid_files_set

    df_filtered = df[df.apply(is_in_folder, axis=1)]
    
    # 保存新的对齐后的 CSV
    df_filtered.to_csv(CSV_OUTPUT, index=False)
    
    print(f"📄 步骤B：CSV 筛选完毕！")
    print(f"   - 原始 CSV 行数：{len(df)}")
    print(f"   - 筛选后 CSV 行数：{len(df_filtered)}")
    print(f"   - 共删除了 {len(df) - len(df_filtered)} 行没有对应图像的空标签。")
    print(f"   - 干净的标签文件已保存为: {CSV_OUTPUT}\n")

    # ================= 步骤 C：反向核对（文件夹里是否有 CSV 里找不到的文件？） =================
    # 提取筛选后 CSV 里的组合对
    csv_set = set(zip(df_filtered['Subject'], df_filtered['Image Data ID']))
    
    # 求差集：在文件夹里，但不在 CSV 里
    orphans = valid_files_set - csv_set
    
    if orphans:
        print(f"⚠️ 【警告】发现 {len(orphans)} 个图像文件在 CSV 中没有对应的标签！")
        print("以下是这些没有标签的“孤儿文件”，建议你将它们从文件夹中移走，否则运行 dataloader 会报错：")
        for subj, img_id in orphans:
            orphan_file = f"hdbet_{subj}_I{img_id}.nii.gz"
            print(f"  ❌ 缺失标签: {orphan_file}")
            
        # (可选) 如果你希望脚本直接帮你把这些没标签的文件删掉，可以取消下面两行的注释：
        # for subj, img_id in orphans:
        #     os.remove(os.path.join(IMG_DIR, f"hdbet_{subj}_I{img_id}.nii.gz"))
        # print("（已自动删除上述孤儿文件）")
    else:
        print("✅ 【完美匹配】文件夹中的所有图像，都在 CSV 中找到了对应的标签！太棒了！")

if __name__ == "__main__":
    main()
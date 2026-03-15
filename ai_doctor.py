import torch
import torch.nn.functional as F
import monai
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, CropForegroundd, NormalizeIntensityd, Resized, ToTensord
from ViT_recipe_for_AD.models.make_models import make_vanilla_model
from openai import OpenAI
import gradio as gr
import yaml
import argparse

# ================= 1. 配置区域 (请根据你的实际情况修改) =================
# 你微调得到的最好的一折的权重路径
MODEL_WEIGHT_PATH = "/root/AD_project/ViT_recipe_for_AD/checkpoints/BEST_MODEL_ADNI1_Finetune_1_ADNI1_mode_finetune_seed_42_fold_4_039_006080.pth.tar"
CONFIG_PATH = "/root/AD_project/ViT_recipe_for_AD/configs/config.yaml"

# 大模型 API 配置
# 如果你用的是 OpenAI，base_url 不需要填；如果是其他模型，请填入他们提供的 base_url
LLM_API_KEY = "sk-xxx"
LLM_BASE_URL = "https://xxx/v1" # 所用模型的接口
LLM_MODEL_NAME = "gpt-3.5-turbo" # 替换为你的大模型名称
# =======================================================================

# 初始化 LLM 客户端
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL if LLM_BASE_URL else None)

def load_vit_model():
    """加载你训练好的 3D ViT 模型"""
    print("⏳ 正在加载 3D ViT 视觉大脑...")
    # 构造 args 伪装命令行输入
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.patch_embed_fun = 'conv3d' # # 修复点：必须和你实际训练时的默认参数一致！而非参照你的 config
    args.patch_size = 16
    args.drop_path = 0.1
    args.attn_p = 0.1
    args.p = 0.1
    args.classes_to_use =['CN', 'AD']
    args.mode = 'finetune'
    args.vit_size = 'base'
    
    with open(CONFIG_PATH, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # 【核心修复点】还原 kfold_train.py 里的半路塞入逻辑！
    cfg['MODEL']['n_classes'] = 2
    cfg['MODEL']['patch_embed_fun'] = args.patch_embed_fun
    cfg['MODEL']['patch_size'] = args.patch_size
    cfg['MODEL']['drop_path_rate'] = args.drop_path  # <--- 就是这行解决了你的 KeyError！
    cfg['MODEL']['attn_p'] = args.attn_p
    cfg['MODEL']['p'] = args.p
    
    model = make_vanilla_model(cfg, args)
    
    # 加载你的权重
    checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location='cpu')
    # 兼容处理：微调保存的可能是完整的 dict，也可能包在 'model' 或 'state_dict' 里
    # 1. 第一层脱壳：找到真正装权重的格子 (兼容 'net', 'state_dict', 'model')
    if 'net' in checkpoint:
        checkpoint_model = checkpoint['net']
    elif 'state_dict' in checkpoint:
        checkpoint_model = checkpoint['state_dict']
    elif 'model' in checkpoint:
        checkpoint_model = checkpoint['model']
    else:
        checkpoint_model = checkpoint
        
    # 2. 第二层脱壳：处理多卡训练 (DataParallel) 自动添加的 'module.' 前缀
    clean_state_dict = {}
    for k, v in checkpoint_model.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v  # 去掉开头的 'module.' (一共7个字符)
        else:
            clean_state_dict[k] = v

    # 3. 把干净的权重塞进模型
    model.load_state_dict(clean_state_dict, strict=True)
        
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    print("✅ 视觉大脑加载完毕！")
    return model, cfg

# 预先加载模型，避免每次点击网页都重新加载
vit_model, config = load_vit_model()

def preprocess_image(nii_path, cfg):
    """使用与训练完全一致的 MONAI 数据流处理新上传的图像"""
    # 注意：这里的参数必须和你训练时 config.yaml 里的一模一样！
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes=cfg["TRANSFORMS"]["orientation"]),
        Spacingd(keys=["image"], pixdim=tuple(cfg["TRANSFORMS"]["spacing"])),
        CropForegroundd(keys=["image"], source_key="image"),
        NormalizeIntensityd(keys=["image"], nonzero=True), # 记得把 nonzero=True 硬编码在这里，防止 key error
        Resized(keys=["image"], spatial_size=tuple(cfg["TRANSFORMS"]["resize"])),
        ToTensord(keys=["image"])
    ])
    
    data = {"image": nii_path}
    data = transforms(data)
    img_tensor = data["image"].unsqueeze(0) # 增加 batch 维度 (1, C, H, W, D)
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    return img_tensor

def generate_medical_report(prediction_class, confidence, patient_age, patient_gender, additional_info):
    """调用大模型生成医疗报告"""
    
    # 构造系统提示词（System Prompt），赋予大模型灵魂
    system_prompt = """
    你是一位世界顶尖的神经内科主任医师，尤其擅长阿尔茨海默病（Alzheimer's Disease, AD）的早期诊断与干预。
    你现在正在使用最先进的 3D 深度学习影像系统辅助诊断。
    请根据深度学习模型提供的核磁共振（MRI）预测结果以及患者的基本信息，出具一份详尽、专业且具有人文关怀的临床分析报告。
    
    报告需包含以下部分：
    1. 患者基本信息摘要
    2. AI 影像分析结果及置信度解读
    3. 临床诊断意见
    4. 进一步检查建议（如量表测试、PET扫描、脑脊液检测等）
    5. 生活方式及医疗干预建议
    
    语气要求：专业严谨、客观中立、对患者及家属展现出同理心和安抚感。不要绝对化（诊断结果仅作辅助参考）。
    """
    
    # 构造给大模型的具体病例信息
    user_prompt = f"""
    【患者基础信息】
    年龄：{patient_age} 岁
    性别：{patient_gender}
    临床备注症状：{additional_info}
    
    【3D ViT 影像 AI 预测结果】
    模型预测类别：{prediction_class} (CN代表认知正常，AD代表阿尔茨海默病)
    AI 判断置信度：{confidence:.2f}%
    
    请根据以上信息，为该患者生成一份专业的神经内科辅助诊断报告。
    """
    
    print("🧠 正在呼叫大语言模型撰写报告...")
    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7, # 稍微控制一点随机性，让报告更严谨
        max_tokens=1500
    )
    
    return response.choices[0].message.content

def ai_doctor_pipeline(nii_file, age, gender, symptoms):
    """处理整个流程：文件上传 -> ViT预测 -> LLM生成报告"""
    if nii_file is None:
        return "请先上传去颅骨后的脑部 MRI 图像 (.nii.gz)"
    
    try:
        # 1. 影像推理
        img_tensor = preprocess_image(nii_file.name, config)
        with torch.no_grad():
            output = vit_model(img_tensor)
            probabilities = F.softmax(output, dim=1).squeeze().cpu().numpy()
        
        # 假设 0 是 CN, 1 是 AD (取决于你 args.classes_to_use 的顺序，通常字母排序 CN 为 0, AD 为 1)
        classes = ['认知正常 (CN)', '阿尔茨海默病 (AD)']
        pred_idx = probabilities.argmax()
        pred_class = classes[pred_idx]
        confidence = probabilities[pred_idx] * 100
        
        # 2. 调用大模型
        report = generate_medical_report(pred_class, confidence, age, gender, symptoms)
        
        return f"### 🔍 影像端侧判定结果\n**模型预测：** {pred_class}\n**AI 置信度：** {confidence:.2f}%\n\n---\n\n### 📝 神经内科主任 AI 综合报告\n" + report
        
    except Exception as e:
        return f"诊断过程中发生错误：{str(e)}"

# ================= 3. Gradio Web 前端界面 =================
print("🌐 正在启动 AI 医生 Web 界面...")

interface = gr.Interface(
    fn=ai_doctor_pipeline,
    inputs=[
        gr.File(label="上传脑部 MRI 图像 (请上传经 HD-BET 去颅骨的 .nii.gz 文件)"),
        gr.Slider(minimum=40, maximum=100, step=1, label="患者年龄", value=70),
        gr.Radio(choices=["男", "女", "其他"], label="患者性别", value="女"),
        gr.Textbox(label="临床症状自述/简短备注", placeholder="例如：近半年来经常忘事，找不到钥匙，有时迷路...", lines=3)
    ],
    outputs=gr.Markdown(label="AI 智能诊断报告"),
    title="🧠 3D-ViT 阿尔茨海默病智能辅助诊断系统",
    description="上传患者的 3D 脑部核磁共振影像，系统将融合 Vision Transformer 的影像预测与大语言模型的临床推理，自动生成诊断与干预建议报告。",
    # theme="huggingface"
)

if __name__ == "__main__":
    # share=True 会生成一个公网链接，即使你在 AutoDL 上跑，也能用你自己的电脑浏览器访问！
    # 只需要指定 6006 端口即可，AutoDL 会自动帮我们做公网穿透
    interface.launch(server_name="0.0.0.0", server_port=6006)
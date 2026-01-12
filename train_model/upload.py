from modelscope.hub.api import HubApi
import os

# === 你的配置信息 ===
YOUR_TOKEN = "ms-a0ddd846-25ce-4e78-bd7a-68d389153978"
YOUR_MODEL_ID = "qinglandesu/GPUclass_qwen1"
LOCAL_MODEL_DIR = "/root/notebook/qinglan/train_model/modelscope/400"
# ==================

print(f"正在登录 ModelScope...")
api = HubApi()
api.login(YOUR_TOKEN)

print(f"准备将 {LOCAL_MODEL_DIR} 上传到 {YOUR_MODEL_ID} ...")

if not os.path.exists(LOCAL_MODEL_DIR):
    print(f"❌ 错误：找不到本地模型文件夹 {LOCAL_MODEL_DIR}")
    print("请确认你之前是否成功运行了 prepare.py 并生成了 modelscope 文件夹。")
else:
    try:
        api.push_model(
            model_id=YOUR_MODEL_ID, 
            model_dir=LOCAL_MODEL_DIR
        )
        print("🎉 恭喜！模型上传成功！")
        print("现在请回到 ModelScope 网页刷新，那个粉红色的警告应该消失了。")
    except Exception as e:
        print(f"❌ 上传失败，错误信息: {e}")

import os
import shutil
from modelscope import snapshot_download

# 定义目标目录
text_model_dir = './local_text_model'
clip_model_dir = './local_clip_model'

print("=== 1. 正在下载文本模型 (all-MiniLM-L6-v2) ===")
# 这个之前下载成功了，为了节省时间，我们检查一下，如果存在就不重复下了
if os.path.exists(text_model_dir) and os.listdir(text_model_dir):
    print(f"✅ 文本模型已存在，跳过下载: {text_model_dir}")
else:
    try:
        src_text = snapshot_download('AI-ModelScope/all-MiniLM-L6-v2')
        shutil.copytree(src_text, text_model_dir)
        print(f"✅ 文本模型下载完毕: {text_model_dir}")
    except Exception as e:
        print(f"❌ 文本模型下载失败: {e}")

print("\n=== 2. 正在下载图像模型 (clip-ViT-B-32) ===")
# 修正点：换用了 ModelScope 上存在的正确 ID
try:
    # 尝试下载 sentence-transformers 官方版
    src_clip = snapshot_download('sentence-transformers/clip-ViT-B-32')

    if os.path.exists(clip_model_dir):
        shutil.rmtree(clip_model_dir)
    shutil.copytree(src_clip, clip_model_dir)
    print(f"✅ 图像模型已准备好: {clip_model_dir}")

except Exception as e:
    print(f"❌ 第一次尝试失败，尝试备用多语言版本... ({e})")
    try:
        # 如果上面那个还不行，下载多语言版（完全兼容，且更强）
        src_clip = snapshot_download('sentence-transformers/clip-ViT-B-32-multilingual-v1')
        if os.path.exists(clip_model_dir):
            shutil.rmtree(clip_model_dir)
        shutil.copytree(src_clip, clip_model_dir)
        print(f"✅ 图像模型(多语言版)已准备好: {clip_model_dir}")
    except Exception as e2:
        print(f"❌ 图像模型下载彻底失败: {e2}")

print("\n=== 所有模型下载完成！ ===")
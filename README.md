# Local Multimodal AI Agent (本地 AI 智能文献与图像管理助手)


## 1. 项目简介 (Project Introduction)

本项目是一个基于 Python 的本地多模态 AI 智能助手，旨在解决本地大量文献和图像素材管理困难的问题。本项目利用多模态神经网络技术，实现了对内容的**语义搜索**和**自动分类**，彻底摆脱了传统文件名搜索的限制。

**核心优势**:
*   **本地化部署**: 所有数据和计算均在本地进行，确保用户隐私安全。
*   **多模态能力**: 集成了文本和图像的理解与检索能力。
*   **智能管理**: 通过语义理解，实现文献的精准搜索和自动分类，以及图片的以文搜图。

## 2. 核心功能列表 (Core Features)

### 2.1 智能文献管理 (Intelligent Document Management)
*   **语义搜索**: 使用自然语言提问，系统返回最相关的论文文件。
*   **自动分类与整理**:
    *   **单文件处理**: 添加新论文时，根据指定主题自动分析内容并归类到对应的子文件夹。
    *   **批量整理**: 一键扫描混乱的文件夹，识别主题并归档所有 PDF 文件。
*   **文件索引**: 为快速定位提供相关文件列表。

### 2.2 智能图像管理 (Intelligent Image Management)
*   **以文搜图**: 通过自然语言描述，在本地图片库中查找最匹配的图像。

## 3. 技术选型 (Technical Stack & Recommendations)

本项目采用模块化设计，选用轻量级且性能优越的模型与工具，确保在本地设备上也能流畅运行。

| 组件           | 选型                               | 理由                                                                                                                   |
| :------------- | :--------------------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| **语言**       | Python 3.10                        | 稳定且生态成熟，对 AI 库支持良好。                                                                                     |
| **文本嵌入**   | `SentenceTransformers` (`all-MiniLM-L6-v2`) | 速度快，内存占用小，语义表征能力强。                                                                                   |
| **图像/跨模态**| `SentenceTransformers` (`clip-ViT-B-32`) | OpenAI 的 CLIP 模型，经典的图文匹配模型，适合 Zero-Shot 图像检索。                                                      |
| **向量数据库** | `ChromaDB`                         | 轻量级、嵌入式数据库，无需配置服务器，支持本地持久化存储。                                                               |
| **PDF 处理**   | `PyPDF2`                           | 用于从 PDF 文件中提取文本内容。                                                                                         |
| **CLI 框架**   | `argparse`                         | Python 内置库，用于创建命令行接口，方便用户交互。                                                                      |
| **模型下载**   | `ModelScope`                       | 采用国内镜像源下载模型，解决网络问题，速度快且稳定。                                                                   |
| **SQLite 兼容**| `pysqlite3-binary` + 代码补丁      | 解决 ChromaDB 对 SQLite 版本的高要求问题，确保数据库正常工作。                                                         |

## 4. 环境配置与依赖安装 (Environment & Installation)

### 4.1 前置要求
*   **操作系统**: Windows / macOS / Linux (本项目在 Linux 下测试通过)
*   **Python 版本**: 建议 **Python 3.10** (项目已配置为 3.10，推荐新建 Conda 环境)
*   **内存**: 建议 8GB 及以上
*   **GPU**: RTX 3060 (12GB/6GB) 或同级别显卡，用于加速模型加载和推理。

### 4.2 安装步骤

1.  **创建并激活 Conda 环境** (推荐使用 Python 3.10)
    ```bash
    conda create -n local_agent python=3.10 -y
    conda activate local_agent
    ```

2.  **安装基础依赖**
    ```bash
    pip install sentence-transformers Pillow PyPDF2 tqdm torch torchvision torchaudio
    ```
    *(注意: `torch` 会自动匹配你的 CUDA 版本，如果 PyTorch 安装不成功，请参考之前的沟通记录，尝试使用 Conda 安装)*

3.  **安装 ChromaDB**
    ```bash
    pip install chromadb
    ```

4.  **修复 SQLite 版本问题 (关键步骤)**
    ```bash
    conda install -c conda-forge sqlite -y
    ```

5.  **下载模型权重**
    *   在项目根目录下（与 `main.py` 同级）创建 `download_all.py` 文件，并将以下内容复制进去：
        ```python
        import os
        import shutil
        from modelscope import snapshot_download

        text_model_dir = './local_text_model'
        clip_model_dir = './local_clip_model'

        print("=== 1. 正在下载文本模型 (all-MiniLM-L6-v2) ===")
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
        try:
            src_clip = snapshot_download('sentence-transformers/clip-ViT-B-32')
            if os.path.exists(clip_model_dir):
                shutil.rmtree(clip_model_dir)
            shutil.copytree(src_clip, clip_model_dir)
            print(f"✅ 图像模型已准备好: {clip_model_dir}")
        except Exception as e:
            print(f"❌ 第一次尝试失败，尝试备用多语言版本... ({e})")
            try:
                src_clip = snapshot_download('sentence-transformers/clip-ViT-B-32-multilingual-v1')
                if os.path.exists(clip_model_dir):
                    shutil.rmtree(clip_model_dir)
                shutil.copytree(src_clip, clip_model_dir)
                print(f"✅ 图像模型(多语言版)已准备好: {clip_model_dir}")
            except Exception as e2:
                print(f"❌ 图像模型下载彻底失败: {e2}")

        print("\n=== 所有模型下载完成！ ===")
        ```
    *   运行下载脚本：
        ```bash
        python download_all.py
        ```
        *(模型文件较大，请耐心等待下载完成)*

6.  **修改 `main.py`**
    *   将 `main.py` 文件中的 `TEXT_MODEL_NAME` 修改为 `'./local_text_model'`。
    *   将 `IMAGE_MODEL_NAME` 修改为 `'./local_clip_model'`。
    *   （可选）在 `main.py` 最开头加入 SQLite 补丁代码，防止再次报错。

## 5. 详细使用说明 (Detailed Usage Instructions)

本项目提供命令行接口，方便用户执行各种操作。确保你在项目根目录下（即 `main.py` 所在目录）运行以下命令。

### 5.1 批量导入并自动分类论文
扫描指定文件夹内的所有 PDF 文件，分析内容，并根据提供的**主题列表**将其移动到相应的子文件夹中，同时建立向量索引。

*   **命令格式**:
    ```bash
    python main.py batch_import <folder_path> --mode paper --topics "Topic1,Topic2,..."
    ```

*   **示例**:
    假设你的论文存放在 `./mypapers` 文件夹，你想按 "Deep Learning", "Mathematics", "History" 等主题分类：
    ```bash
    python main.py batch_import "./mypapers" --mode paper --topics "Deep Learning,Mathematics,History,Literature"
    ```
    运行后，系统会在 `./mypapers` 目录下创建如 `Deep Learning/`、`History/` 等文件夹，并将匹配的论文移动进去。

### 5.2 批量导入本地图片库
扫描指定文件夹内的图片文件，并为它们建立以文搜图的索引。

*   **命令格式**:
    ```bash
    python main.py batch_import <folder_path> --mode image
    ```

*   **示例**:
    导入 `./myimages` 文件夹中的所有图片：
    ```bash
    python main.py batch_import "./myimages" --mode image
    ```
    图片文件将不会被移动。

### 5.3 语义搜索论文
使用自然语言查询，系统将在已索引的论文库中查找最相关的文档。

*   **命令格式**:
    ```bash
    python main.py search_paper "<Your Query>"
    ```

*   **示例**:
    搜索关于 Transformer 架构的论文：
    ```bash
    python main.py search_paper "What is the transformer architecture?"
    ```
    系统会返回最相关的论文文件名、路径和相关性得分。

### 5.4 以文搜图
通过自然语言描述，在本地图片库中查找最匹配的图像。

*   **命令格式**:
    ```bash
    python main.py search_image "<Your Image Description>"
    ```

*   **示例**:
    搜索一张日落的照片：
    ```bash
    python main.py search_image "a photo of a sunset"
    ```
    系统会列出匹配的图片文件名和路径。

## 6. 项目结构 (Project Structure)

```text
.
├── main.py              # 核心逻辑与命令行入口
├── download_all.py      # 用于下载模型权重的脚本
├── requirements.txt     # 项目依赖 (可手动创建)
├── README.md            # 项目说明文档
├── local_text_model/    # all-MiniLM-L6-v2 模型权重 (自动生成)
├── local_clip_model/    # clip-ViT-B-32 模型权重 (自动生成)
├── local_knowledge_db/  # ChromaDB 向量数据库存储目录 (自动生成)
└── ...                  # your_papers/ , your_images/ 等数据文件夹

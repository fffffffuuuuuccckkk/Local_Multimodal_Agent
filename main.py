# --- [关键修复] 强制替换系统 SQLite (针对 ChromaDB 版本报错问题) ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# -------------------------------------------------------------

import os
import shutil
import argparse
import warnings
from typing import List

# 忽略部分库的警告信息，保持输出整洁
warnings.filterwarnings("ignore")

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import PyPDF2
from tqdm import tqdm
import torch

# ================= 配置区域 =================
# 向量数据库路径
DB_PATH = "./local_knowledge_db"

# [修改点] 指向你刚才下载的【本地模型文件夹】
# 这样程序就不会去 HuggingFace 联网下载了
TEXT_MODEL_NAME = './local_text_model'
IMAGE_MODEL_NAME = './local_clip_model'

class LocalAIAgent:
    def __init__(self):
        print("正在初始化 AI Agent...")
        print(f"Loading Text Model from: {TEXT_MODEL_NAME}")
        print(f"Loading Image Model from: {IMAGE_MODEL_NAME}")

        # 1. 初始化嵌入模型 (直接加载本地文件)
        try:
            # 用于论文语义搜索和分类
            self.text_model = SentenceTransformer(TEXT_MODEL_NAME)
            # 用于以文搜图 (加载 CLIP)
            self.clip_model = SentenceTransformer(IMAGE_MODEL_NAME)
        except Exception as e:
            print(f"\n[严重错误] 模型加载失败: {e}")
            print("请检查当前目录下是否有 'local_text_model' 和 'local_clip_model' 文件夹。")
            print("如果没有，请先运行 download_all.py 下载模型。\n")
            raise e

        # 2. 初始化 ChromaDB 向量数据库
        self.client = chromadb.PersistentClient(path=DB_PATH)

        # 创建/获取集合
        # 论文集合 (使用 L2 距离)
        self.paper_collection = self.client.get_or_create_collection(
            name="papers",
            metadata={"hnsw:space": "cosine"}
        )
        # 图像集合
        self.image_collection = self.client.get_or_create_collection(
            name="images",
            metadata={"hnsw:space": "cosine"}
        )
        print("初始化完成！")

    def _extract_text_from_pdf(self, pdf_path: str, max_pages: int = 2) -> str:
        """从 PDF 提取文本，默认只读前两页（通常包含标题和摘要）以提高速度"""
        text = ""
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)
                for i in range(min(num_pages, max_pages)):
                    page = reader.pages[i]
                    text += page.extract_text() or ""
        except Exception as e:
            print(f"读取 PDF 失败 {pdf_path}: {e}")
        return text.strip()

    def add_paper(self, file_path: str, topics: str = None):
        """添加论文并在指定 topics 时自动分类"""
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 不存在")
            return

        print(f"正在处理论文: {os.path.basename(file_path)}...")

        # 1. 提取文本
        content = self._extract_text_from_pdf(file_path)
        if not content:
            print("警告: 无法从 PDF 中提取文本，跳过。")
            return

        # 2. 生成向量
        embedding = self.text_model.encode(content).tolist()

        # 3. 存入数据库
        file_id = os.path.abspath(file_path)
        self.paper_collection.upsert(
            documents=[content[:1000]], # 存储前1000字符用于预览
            embeddings=[embedding],
            metadatas=[{"path": file_path, "filename": os.path.basename(file_path)}],
            ids=[file_id]
        )
        print("论文已索引。")

        # 4. 自动分类逻辑 (如果提供了 topics)
        if topics:
            topic_list = [t.strip() for t in topics.split(',')]
            # 计算 content 和 topic_list 的相似度
            topic_embeddings = self.text_model.encode(topic_list)
            # 这里的 embedding 是论文的向量
            # 计算余弦相似度
            similarities = util.cos_sim(embedding, topic_embeddings)[0]

            # 找到最匹配的 Topic
            best_idx = torch.argmax(similarities).item()
            best_topic = topic_list[best_idx]
            score = similarities[best_idx].item()

            print(f"分析主题... 最匹配: '{best_topic}' (置信度: {score:.2f})")

            # 移动文件
            target_dir = os.path.join(os.path.dirname(file_path), best_topic)
            os.makedirs(target_dir, exist_ok=True)
            new_path = os.path.join(target_dir, os.path.basename(file_path))

            try:
                shutil.move(file_path, new_path)
                print(f"文件已归档至: {new_path}")
                # 更新数据库中的路径（需要删除旧的插入新的，或更新 metadata，这里简化处理不更新ID，仅更新metadata）
                self.paper_collection.update(
                    ids=[file_id],
                    metadatas=[{"path": new_path, "filename": os.path.basename(new_path)}]
                )
            except Exception as e:
                print(f"移动文件失败: {e}")

    def add_image(self, image_path: str):
        """添加图片到索引"""
        if not os.path.exists(image_path):
            print(f"文件不存在: {image_path}")
            return

        try:
            img = Image.open(image_path)
            # CLIP 编码图片
            embedding = self.clip_model.encode(img).tolist()

            file_id = os.path.abspath(image_path)
            self.image_collection.upsert(
                embeddings=[embedding],
                metadatas=[{"path": image_path, "filename": os.path.basename(image_path)}],
                ids=[file_id]
            )
            print(f"图片已索引: {os.path.basename(image_path)}")
        except Exception as e:
            print(f"处理图片失败 {image_path}: {e}")

    def search_paper(self, query: str, top_k: int = 3):
        """语义搜索论文"""
        print(f"正在搜索论文: '{query}' ...")
        query_embedding = self.text_model.encode(query).tolist()

        results = self.paper_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        if not results['ids'][0]:
            print("未找到相关论文。")
            return

        print(f"\n====== 找到相关论文 (Top {top_k}) ======")
        for i, meta in enumerate(results['metadatas'][0]):
            distance = results['distances'][0][i] # 距离越小越相似(Cosine distance)
            # 因为我们在 metadata 中用了 cosine 距离，chroma 返回的是 cosine distance (1 - similarity)
            # 或者直接看相对排序
            print(f"[{i+1}] {meta['filename']}")
            print(f"    路径: {meta['path']}")
            print(f"    相关性得分 (距离): {distance:.4f}\n")

    def search_image(self, query: str, top_k: int = 3):
        """以文搜图"""
        print(f"正在以文搜图: '{query}' ...")
        # CLIP 编码文本
        query_embedding = self.clip_model.encode(query).tolist()

        results = self.image_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        if not results['ids'][0]:
            print("未找到相关图片。")
            return

        print(f"\n====== 找到相关图片 (Top {top_k}) ======")
        for i, meta in enumerate(results['metadatas'][0]):
            print(f"[{i+1}] {meta['filename']}")
            print(f"    路径: {meta['path']}")
            # 可以在这里尝试用系统默认查看器打开图片
            # os.startfile(meta['path']) if os.name == 'nt' else ...
            print("")

    def batch_process(self, folder_path: str, topics: str = None, mode="paper"):
        """批量处理文件夹"""
        if not os.path.exists(folder_path):
            print("文件夹不存在")
            return

        print(f"正在扫描目录: {folder_path}")
        files = os.listdir(folder_path)

        for f in tqdm(files):
            full_path = os.path.join(folder_path, f)
            if not os.path.isfile(full_path):
                continue

            ext = f.lower().split('.')[-1]

            if mode == "paper" and ext == "pdf":
                self.add_paper(full_path, topics)
            elif mode == "image" and ext in ['jpg', 'jpeg', 'png', 'bmp']:
                self.add_image(full_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="本地多模态 AI 智能助手")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 1. 添加/整理论文
    parser_add = subparsers.add_parser("add_paper", help="添加或分类单个论文PDF")
    parser_add.add_argument("path", type=str, help="PDF文件路径")
    parser_add.add_argument("--topics", type=str, default=None, help="分类主题，用逗号分隔，如 'CV,NLP'")

    # 2. 批量处理
    parser_batch = subparsers.add_parser("batch_import", help="批量导入文件夹中的文件")
    parser_batch.add_argument("folder", type=str, help="文件夹路径")
    parser_batch.add_argument("--mode", type=str, choices=["paper", "image"], default="paper", help="模式: paper 或 image")
    parser_batch.add_argument("--topics", type=str, default=None, help="[仅paper模式] 分类主题")

    # 3. 搜索论文
    parser_search_p = subparsers.add_parser("search_paper", help="语义搜索论文")
    parser_search_p.add_argument("query", type=str, help="搜索问题，如 'Transformer architecture'")

    # 4. 以文搜图
    parser_search_i = subparsers.add_parser("search_image", help="以文搜图")
    parser_search_i.add_argument("query", type=str, help="图片描述，如 'a cat on the grass'")

    args = parser.parse_args()

    if args.command:
        agent = LocalAIAgent()
        if args.command == "add_paper":
            agent.add_paper(args.path, args.topics)
        elif args.command == "batch_import":
            agent.batch_process(args.folder, args.topics, args.mode)
        elif args.command == "search_paper":
            agent.search_paper(args.query)
        elif args.command == "search_image":
            agent.search_image(args.query)
    else:
        parser.print_help()
import os
import json
import numpy as np
import faiss
import openai

class FaissMemory:
    def __init__(self, index_path="memory/memory.index"):
        self.memories = []
        self.embeddings = []
        self.index_path = index_path
        self.index = None
        self.embedding_dim = 1536  # OpenAI text-embedding-3-small/large
        self.load_index()

    def get_embedding(self, text):
        # 使用 OpenAI API 取得 embedding
        try:
            response = openai.embeddings.create(
                input=[text],
                model="text-embedding-3-small"  # 你也可以改成 text-embedding-3-large
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            print(f"取得 embedding 失敗：{e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def add(self, text):
        emb = self.get_embedding(text)
        self.memories.append(text)
        self.embeddings.append(emb)
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(np.array([emb], dtype=np.float32))
        self.save_index()

    def build_index(self):
        if not self.embeddings:
            return
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(np.array(self.embeddings, dtype=np.float32))

    def save_index(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            # 同步存文字記憶
            with open(self.index_path + ".txt", "w", encoding="utf-8") as f:
                for m in self.memories:
                    f.write(m + "\n")

    def load_index(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            # 載入文字記憶
            txt_path = self.index_path + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    self.memories = [line.strip() for line in f if line.strip()]
        else:
            self.index = None

    def search_semantic(self, query, topk=5):
        if self.index is None or not self.memories:
            return []
        q_emb = self.get_embedding(query)
        D, I = self.index.search(np.array([q_emb], dtype=np.float32), topk)
        return [self.memories[i] for i in I[0] if i < len(self.memories)]

    def import_memories(self, path):
        new_memories = []
        if path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        new_memories.append(line)
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data:
                if isinstance(entry, dict):
                    text = entry.get("text")
                else:
                    text = entry
                if text:
                    new_memories.append(text)
        # 清空舊記憶，導入新記憶
        self.memories = []
        self.embeddings = []
        for m in new_memories:
            self.memories.append(m)
            self.embeddings.append(self.get_embedding(m))
        self.build_index()
        self.save_index()

    def export_memories(self, export_path="memory/export.txt"):
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        with open(export_path, "w", encoding="utf-8") as f:
            for m in self.memories:
                f.write(m + "\n")

# 單例
faiss_memory = FaissMemory()
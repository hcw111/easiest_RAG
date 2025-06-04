"""
实现【向量检索】和【关键词检索】的混合检索模块
1、向量检索
将文本块向量化 并存入向量数据库
2、关键词检索
将文本块分词，提取关键词，并存入bm25
3、混合检索
根据两种检索内容，进行去重，合并
"""
import os
import logging
import jieba
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
logging.basicConfig(level=logging.INFO)

# 初始化嵌入模型
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)


# 向量存储以及检索
class VectorIndexManager:
    """基于chroma向量数据库的向量检索模块"""
    def __init__(self, choice_db):
        # 接收向量数据库实例，并初始化
        self.clint = choice_db
        self.db = self.clint.create_collection(name="store_vec")

    def vector_clear(self):
        """清空向量数据库"""
        # 检查此时数据库中有哪些集合
        db_name = self.clint.list_collections()
        # 如果不为空，说明还有数据待删除
        if db_name:
            self.clint.delete_collection("store_vec")
        else:
            logging.info("数据库已空")

    def vector_store(self, documents, doc_ids, doc_metadata):
        """将文本转化为向量，并存储"""
        if len(documents) == 0:
            logging.error("文本块为空")
            return

        try:
            vector = EMBED_MODEL.encode(documents)
            vector_np = np.array(vector).astype('float32')
            self.db.add(
                embeddings=vector_np,
                metadatas=doc_metadata,
                documents=documents,
                ids=doc_ids,
            )
            logging.info(f"向量化存储完成,文本块数量为{len(documents)}")
        except Exception as e:
            logging.error(f"向量化或向量存储失败:{str(e)}")

    def vector_retrieval(self, query, top_k=5):
        """向量相似度查询，返回查询结果"""
        if not self.db:
            logging.error("向量数据库为空")
            return []

        try:
            logging.info(f"开始向量化检索，查询内容为{query}")

            embedding_query = EMBED_MODEL.encode([query])
            db_results = self.db.query(
                query_embeddings=embedding_query,
                n_results=top_k
            )

            # 格式化返回结果
            try:
                results = [
                    {
                        "id": db_results["ids"][0][i],
                        "content": db_results["documents"][0][i],
                        "distances": float(db_results["distances"][0][i]),
                        "metadata": db_results["metadatas"][0][i]
                    }
                    for i in range(len(db_results["ids"][0]))]
                logging.info(f"向量检索成功，返回结果数量为{len(results)}")
                return results
            except Exception as e:
                logging.error(f"向量检索格式化失败:{str(e)}")
                return db_results

        except Exception as e:
            logging.error(f"向量检索失败:{str(e)}")
            return []


# 关键词存储及检索
class BM25IndexManager:
    def __init__(self):
        self.bm25_index = None
        self.num_map_id = {}
        self.tokenized_corpus = []
        self.text_corpus = []
        self.metadata_list = []

    def build_index(self, document, doc_ids, metadata_list):
        """构建关键词索引"""
        if not document:
            logging.error("文本块为空，无法构建BM25关键词索引")
            return

        self.text_corpus = document
        self.metadata_list = metadata_list
        self.num_map_id = {i: doc_id for i, doc_id in enumerate(doc_ids)}
        # 用jieba，对文本进行分词，并将分词后的结果构建BM25索引
        self.tokenized_corpus = [list(jieba.cut(doc)) for doc in document]
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        if self.bm25_index:
            logging.info(f"BM25索引构建完成，索引数量为{len(self.tokenized_corpus)}")
        else:
            logging.error("BM25索引构建失败")

    def bm25_clear(self):
        """清空索引"""
        self.bm25_index = None
        self.num_map_id = {}
        self.tokenized_corpus = []
        self.text_corpus = []

    def bm25_search(self, query, top_k=5):
        """关键词检索"""
        if not self.bm25_index:
            logging.error("BM25索引为空")
            return []
        logging.info(f"开始关键词检索，查询内容为{query}")
        query_tokens = list(jieba.cut(query))
        scores = self.bm25_index.get_scores(query_tokens)

        top_indexes = np.argsort(scores)[::-1][:top_k]  # 返回的是分数最高的数字索引，再通过num_map_id映射回原始ID

        # 将索引转换为原始ID, 并格式化返回结果
        results = []
        for index in top_indexes:
            if scores[index] > 0:
                results.append({
                    "id": self.num_map_id[index],
                    "content": self.text_corpus[index],
                    "score": float(scores[index]),
                    "metadata": self.metadata_list[index]
                })  # 格式化返回结果
        return results


def merge_results(vector_results, keyword_results, alpha=0.8):
    """融合向量检索结果和关键词检索结果
    :param vector_results: 向量检索结果，结构为[{'ids':id, 'content': content, 'metadata': metadata, 'distances': distances}, ...]
    :param keyword_results: 关键词检索结果，结构为[{'ids':id, 'content': content, 'score': score}, ...]
    :param alpha: 语义搜索权重[0-1]
    :return 合并后的结果[(doc_id, {'score': score, 'content': content, 'metadata': metadata}),....]
    """
    # 创建一个字典来存储合并后的结果
    merged_results = {}
    logging.info(f"开始融合向量检索结果和关键词检索结果，向量检索结果数量为{len(vector_results)}, 关键词检索结果数量为{len(keyword_results)}")
    # 处理向量检索结果
    if vector_results:
        max_distance = max([item['distances'] for item in vector_results])
        for item in vector_results:
            doc_id = item['id']
            content = item['content']
            distance = item['distances']

            # 将距离转换为分数（假设距离越小相似度越高）
            score = 1 - distance / max(1, max_distance)
            merged_results[doc_id] = {'score': alpha * score, 'content': content, 'metadata': item['metadata']}
    else:
        logging.warning("检索结果合并，向量检索结果为空")

    # 处理关键词检索结果
    if keyword_results:
        max_score = max([item['score'] for item in keyword_results])
        for item in keyword_results:
            doc_id = item['id']
            content = item['content']
            score = item['score']
            metadata = item['metadata']
            normalize_score = score / max_score
            if doc_id in merged_results:
                # 如果文档ID已经存在，则叠加分数
                merged_results[doc_id]['score'] += (1 - alpha) * normalize_score
            else:
                # 否则直接赋值
                merged_results[doc_id] = {'score': (1 - alpha) * normalize_score, 'content': content, 'metadata': metadata}
    else:
        logging.warning("检索结果合并，关键词检索结果为空")

    # 转换为列表并按分数排序
    merged_results = sorted(merged_results.items(), key=lambda x: x[1]['score'], reverse=True)
    return merged_results


if __name__ == "__main__":
    import doc_process

    # 初始化向量数据库
    client = chromadb.Client()
    collection = client.create_collection(name="store_vec")
    chunks, ids, metadata = doc_process.multiple_file_process(["C:\\Users\\hcwam\\Desktop\\明朝那些事儿.pdf"])

    vector = VectorIndexManager(choice_db=collection)
    vector.vector_store(chunks, ids, metadata)
    bm25 = BM25IndexManager()
    bm25.build_index(chunks, ids, metadata)
    print(merge_results(vector.vector_retrieval("朱元璋哪一年去世的？"), bm25.bm25_search("朱元璋哪一年去世的？")))







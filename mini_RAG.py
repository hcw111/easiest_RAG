"""
整合所有RAG函数，实现RAG
"""

import logging
import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from doc_process import multiple_file_process
from vector_keyword_store import VectorIndexManager, BM25IndexManager, merge_results

# 加载模型环境变量
load_dotenv()
TONGYI_KEY = os.getenv("DASHSCOPE_API_KEY")
MODEL = OpenAI(api_key=TONGYI_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = "qwen3-235b-a22b"

# 过滤日志
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)


def chat(query):
    chat_model = MODEL.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": query}],
        extra_body={"enable_thinking": False},
        stream=True,

    )
    # for chunk in chat_model:
    #     if chunk.choices[0].delta.content:
    #         print(chunk.choices[0].delta.content, end="")

    return chat_model


def get_answer(query, retriever_results):
    if not retriever_results:
        logging.error("没有检索到相关知识，知识库为空")
        return

    # 用检索到的知识构建上下文
    context_with_llm = []

    for doc_id, item in retriever_results:
        content = item.get('content', "")
        meta = item.get('metadata', {})
        source = meta.get('source', "")
        context_with_llm.append(f"[文档名:{source}]\n{content}")

    context = "\n".join(context_with_llm)

    # 构建prompt
    prompt_template = """
    现在你是一个知识库问答助手，你的名字是“超级无敌究极智能问答助手”，请根据提供的知识库内容，回答用户的问题。
    
    知识库内容：
    {context}
    
    用户问题：
    {query}
    
    请遵循以下回答原则：
    1. 仅基于提供的参考内容回答问题，不要使用你自己的知识
    2. 如果参考内容中没有足够信息，请坦诚告知你无法回答
    3. 回答应该全面、准确、有条理，并使用适当的段落和结构
    4. 请用中文回答
    5. 在回答末尾标注信息来源{document_source}
    
    请开始回答：
    """

    prompt = prompt_template.format(context=context,
                                    query=query,
                                    document_source='，如果有多个文档，列出用到了那些文档的内容')

    # 调用模型
    try:
        chat_model = MODEL.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}],
            extra_body={"enable_thinking": False},
            stream=True,
        )
        return chat_model
    except Exception as e:
        logging.error(f"回答失败，错误信息：{str(e)}")
        return "回答失败，请重试"


class RAG:
    # 初始化数据库
    def __init__(self):
        db = chromadb.Client()
        self.vector = VectorIndexManager(choice_db=db)
        self.bm25 = BM25IndexManager()
        self.documents, self.doc_ids, self.metadata = [], [], []

    # 文件处理
    def process_file(self, file_paths: list):
        self.documents, self.doc_ids, self.metadata = multiple_file_process(file_paths)
        return self.documents, self.doc_ids, self.metadata

    #  向量存储
    def store_vector(self):
        self.vector.vector_store(self.documents, self.doc_ids, self.metadata)
        self.bm25.build_index(self.documents, self.doc_ids, self.metadata)

    # 检索知识
    def query_retriever(self, query):
        vector_results = self.vector.vector_retrieval(query)
        bm25_results = self.bm25.bm25_search(query)
        return merge_results(vector_results, bm25_results)

    # 检索并返回答案
    def query_answer(self, query):
        retriever_results = self.query_retriever(query)
        # get_answer函数中已经设置为空检测，所以这里不需要再做检测
        return get_answer(query, retriever_results)

    # 清理知识库
    def clear_vector(self):
        self.vector.vector_clear()
        self.bm25.bm25_clear()


if __name__ == "__main__":
    rag = RAG()
    while True:
        # 接受文件，处理文件
        print("\n分批次输入文件路径，\n输入格式为：D:/a/b,则处理当前文件并等待处理下一个文件；\n若输入1，代表文件上传完毕，可开始检索聊天；\n输入0结束问答程序")
        file = input("\n请输入你的文件路径：")
        # 输入0结束整个程序
        if file == "0":
            break
        # 输入1结束文件上传，开始检索聊天
        elif file == "1":
            while True:
                query = input("\n请输入问题（输入数字0跳出问题询问）：")
                if query == "0":
                    break
                result = rag.query_answer(query)
                if result:
                    for chunk in result:
                        if chunk.choices[0].delta.content:
                            print(chunk.choices[0].delta.content, end="")
        # 如果输入不是0，也不是1，则输入文件路径，处理文件
        else:
            documents, doc_ids, metadata = rag.process_file([file])
            rag.store_vector()
            if not documents:
                print(f"{file}文件处理失败，请重新输入")
            continue

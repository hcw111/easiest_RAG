<div align="center">
<h1> 基于本地文档的智能问答系统 </h1>
<p>
<img src="https://img.shields.io/badge/python-3.10%2b-blue.svg" alt="python">
<img src="https://img.shields.io/badge/license-MIT-green.svg" alt="license">
<img src="https://img.shields.io/badge/RAG-PDF%2bWord%2bPPT..-lightgrey.svg" alt="RAG">
<img src="https://img.shields.io/badge/LLM-Qwen3-orange.svg" alt="LLM">
<img src="https://img.shields.io/badge/UI-Gradio-yellow.svg" alt="embedding">
<img src="https://img.shields.io/badge/Database-Chromadb-blueviolet.svg" alt="embedding">
</p>
**最简单的RAG系统，模块化各组件，帮助你快速理解RAG的核心功能**
</div>

## 项目介绍
根据RAG的核心功能，将本地文档进行向量化，并保存到数据库中，然后使用大模型进行问答。
* **简要介绍** 本项目主要是为了初学者能对RAG系统的各个组件进行理解，因此将整体的RAG系统进行模块化，方便初学者理解RAG的核心功能。
* **简化项目** 相比较其余项目，本项目的RAG系统更加简单，只支持本地文档进行问答。
* **项目后续** 为了初学者学习，此项目后续不会增加过多复杂的功能。但后续会新开项目，做一个功能更多的RAG系统，支持网页搜索或现有数据库进行问答。

## 快速开始
1. 安装依赖包
```
pip install -r requirements.txt
```
2. 大模型配置
* 我用的是阿里云的API，可以自行去阿里云官网申请，申请和充值都很简单，方便后续长期使用
* 配置方法一
```angular2html
在文件.env中配置大模型API密钥
```
* 配置方法二
```angular2html
在系统环境变量中配置大模型API密钥
DASHSCOPE_API_KEY=sk-****
```
3. 快速体验
```
python run.py
启动WebUI进行问答

python miniRAG.py
不启用UI，在python终端进行问答
```
###### 运行后，访问地址：http://127.0.0.1:7860/ ，即可在网页UI中进行使用

## 功能详情
### 文档读取
在文件process_doc.py中，根据根据文档类型，进行文档读取，并返回文档内容。现在支持常见的文档类型，如PDF、Word、PPT、Excel等。
输入单/多个文件路径，返回所有文档内容
### 数据分块，向量化，分词，存储
在文件vector_keyword_store.py中，根据文档内容进行分块，将文本块向量化并储存在chroma数据库，将文本块分词并储存在BM25索引中。将两者检索后的内容合并为最终结果。
输入单/多个文档内容以及查询（query），返回检索到的文本块，并格式化输出。
### 检索，并由大模型生成答案
在文件miniRAG.py中，根据用户输入的问题，检索出最匹配的文本块，并使用大模型进行问答。
输入问题（query），返回检索到的文本块，格式化prompt，输入大模型进行问答。
### UI
在文件run.py中，使用Gradio进行UI展示，用户输入问题，点击按钮，即可获得答案。记录处理过程中的log日志。


## 后续安排
本人也是初学者，后续会新开项目，做一个功能更多的RAG系统，支持网页搜索或现有的数据库进行问答。
1. * **分块** 更精细的划分模块，支持多种划分模式。如：基于语义分块，基于结构分块，基于内容分块等。
2. * **数据库** 更多类型的数据库支持，以及混合检索策略。如：Milvus、FAISS等。
3. * **大模型** 支持市面上常见的大模型。如：ChatGLM、ChatGPT等。
4. * **数据** 更多的数据来源，支持网页搜索、数据库、文件、外部API等。
5. * **重排序** 对检索结果进行重排序，提升大模型回答效果。如：基于CrossEncoder、TF-IDF、Cosine Similarity等。
6. * **多模态检索** 支持图片，音频等非文本的检索。

### 后续安排依然会选择所有功能模块化，方便阅读理解。而不是将各种功能放在一个文件，导致阅读困难



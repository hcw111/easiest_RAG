import os
import gradio as gr
import logging
import requests
from io import StringIO
import mini_RAG

rag = mini_RAG.RAG()


# 自定义日志处理器
class StringLoggerHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__()
        self.log_stream = StringIO()
        self.setStream(self.log_stream)

    def emit(self, record):
        super().emit(record)
        self.log_stream.write('\n')

    def get_logs(self):
        return self.log_stream.getvalue()

    def clear_logs(self):
        self.log_stream.truncate(0)
        self.log_stream.seek(0)


# Gradio 前端构建
def build_gradio_app():
    string_handler = StringLoggerHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    string_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(string_handler)
    # 过滤日志，以下日志不显示
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    def upload_document(files, url):
        string_handler.clear_logs()
        if files is None and not url:
            return "请上传文件或输入URL", "", string_handler.get_logs()

        file_paths = []
        file_names = []
        if files:
            for f in files:
                if f is not None:
                    file_paths.append(f.name)
                    file_names.append(os.path.basename(f.name))

        try:
            if file_paths:
                documents, doc_ids, doc_metadata = rag.process_file(file_paths)
                rag.store_vector()
                logging.info("文档处理完成")
                return "文档上传成功并处理完成", "\n".join(documents[:5]), string_handler.get_logs(), file_names
            elif url:
                logging.info(f"正在抓取网页内容: {url}")
                data = {"url": url}
                response = requests.post("http://localhost:8000/upload_url", json=data)
                if response.status_code == 200:
                    result = response.json()
                    logging.info("网页内容抓取成功")
                    return result["message"], result["content_preview"], string_handler.get_logs(), file_names
                else:
                    logging.error(f"网页抓取失败: {response.status_code}")
                    return f"错误：{response.status_code}", "", string_handler.get_logs(), file_names
            else:
                return "请上传文件或输入URL", "", string_handler.get_logs(), file_names
        except Exception as e:
            logging.error(f"发生异常：{str(e)}", exc_info=True)
            return f"发生异常：{str(e)}", "", string_handler.get_logs(), file_names

    def clear_data():
        """清理数据库中的所有消息，适合在更换数据库之前操作"""
        string_handler.clear_logs()
        try:
            logging.info("正在清除数据和向量库...")
            rag.vector.vector_clear()
            rag.bm25.bm25_clear()
            logging.info("数据清除成功")
            return string_handler.get_logs(), ''
        except Exception as e:
            logging.error(f"清除失败：{str(e)}", exc_info=True)
            return string_handler.get_logs(), ''

    def clear_log_data():
        """清理日志窗口的输出数据"""
        return

    def respond(message, chat_history):
        """gradio消息回复函数"""
        full_response = ""
        if chat_history is None:
            chat_history = []

        chat_history.append((message, ""))
        llm_out = rag.query_answer(message)
        # llm_out = mini_RAG.chat(message)

        # 检测输出是否为空，则返回错误信息
        if not llm_out:
            chat_history[-1] = (message, "未检测到数据，请检查数据是否上传！！")
            yield "", chat_history, string_handler.get_logs()
        # 若不为空，则流式返回输出信息
        else:
            for chunk in llm_out:
                content = chunk.choices[0].delta.content
                full_response += content
                chat_history[-1] = (message, full_response)
                yield "", chat_history, string_handler.get_logs()

    # ================== Gradio 界面 ==================
    with gr.Blocks(title="文档检索聊天系统") as demo:
        gr.Markdown("# 📄 文档检索聊天系统\n上传文档或输入网页URL，然后与AI助手进行对话")

        with gr.Row():
            # 左侧：文档上传
            with gr.Column(scale=1):
                with gr.Tab("文档上传"):
                    file_input = gr.File(label="上传本地文档", file_count="multiple")
                    url_input = gr.Textbox(label="或输入网页URL", placeholder="https://example.com/document.pdf", lines=2)
                    upload_btn = gr.Button("上传并解析")
                    clear_btn = gr.Button("清除上传的数据", variant="secondary")
                    output_msg = gr.Textbox(label="操作结果")
                    preview = gr.Textbox(label="文档内容预览", lines=5, interactive=False)

            # 中间：聊天交互
            with gr.Column(scale=2):
                with gr.Tab("聊天交互"):
                    chatbot = gr.Chatbot(height=450, label="聊天记录")
                    msg = gr.Textbox(label="输入消息")
                    send_chat = gr.Button("发送")

            # 右侧：日志
            with gr.Column(scale=1):
                with gr.Tab("日志"):
                    log_output = gr.Textbox(label="系统日志", lines=20, interactive=False)
                    clear_log_data_btn = gr.Button("清除日志消息", variant="secondary")
                    db_detail = gr.Textbox(label="数据库详情", interactive=False)

        # 绑定事件
        # 点击上传文件或URL，并处理文件
        upload_btn.click(
            fn=upload_document,
            inputs=[file_input, url_input],
            outputs=[output_msg, preview, log_output, db_detail]
        )
        # 点击清除按钮，清除数据库中的所有数据
        clear_btn.click(
            fn=clear_data,
            inputs=[],
            outputs=[log_output, db_detail]
        )
        # 点击发送按钮，处理用户输入
        send_chat.click(
            fn=respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, log_output]
        )
        # 点击清除日志按钮，清除日志消息
        clear_log_data_btn.click(
            fn=clear_log_data,
            inputs=[],
            outputs=[log_output]
        )

    return demo


# 启动应用
if __name__ == "__main__":
    app = build_gradio_app()
    app.launch(share=False)

import os
import queue
import threading
import gradio as gr

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

WEBUI_TITLE = '# OpsGPT'

# 创建新任务
# 查询当前任务状态: 忙碌 / 空闲
# 终止当前任务
# 给定输入，继续当前任务 (直接通过 Queue 给出)

request_queue = queue.Queue()
response_queue = queue.Queue()

os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.com.cn/v1"
os.environ["OPENAI_API_KEY"] = "sk-PRJRgGWFxGpRzO3MyxfTPbUh14HcDGii5geSVfm4ImI46ykh"
    
class CustomizedCallbackHandler(StreamingStdOutCallbackHandler):
    res: queue.Queue

    def __init__(self, res):
        self.out = res

    def on_llm_start(
        self, serialized, prompts, **kwargs
    ) -> None:
        self.out.put(['new_msg'])

    def on_llm_new_token(self, token, **kwargs) -> None:
        self.out.put(['delta', token])

class Task(threading.Thread):
    req: queue.Queue
    res: queue.Queue
    input: str

    def __init__(self, input, req, res):
        super().__init__()
        self.input = input
        self.req = req
        self.res = res
        # llm_0 = OpenAI(streaming=True, callbacks=[CustomizedCallbackManager(self.res)], temperature=0)
        llm = ChatOpenAI(model_name='gpt-4', streaming=True, callbacks=[CustomizedCallbackHandler(self.res)], temperature=0)
        
        def get_input() -> str:
            self.res.put(['end'])
            return self.req.get()
        
        tools = load_tools(
            ["human"], 
            llm=llm,
            input_func=get_input,
        )
        self.agent_chain = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

    def run(self):
        self.agent_chain.run(self.input)
        self.res.put(['end'])

class TaskController(threading.Thread):
    cur_req: queue.Queue
    cur_task: threading.Thread

    def __init__(self):
        super().__init__()
        self.cur_req = None
        self.cur_task = None

    def launch(self, input, task_type: str, task_args=None):
        self.cur_req = queue.Queue()
        task = Task(input, self.cur_req, response_queue)
        self.cur_task = task
        task.start()

    def proceed(self, msg: str):
        self.cur_req.put(msg)

    def cancel(self):
        pass

    def query_status(self):
        response_queue.put(['status', 'IDLE' if self.cur_task is None or not self.cur_task.is_alive() else 'BUSY'])

    def run(self):
        while True:
            item = request_queue.get()
            print(item)
            cmd, *args = item
            if hasattr(self, cmd):
                getattr(self, cmd)(*args)

controller = TaskController()
controller.start()

def get_answer(query, history, mode):
    history.append([query, None])
    yield history, ""
    request_queue.put(['query_status'])
    if response_queue.get()[1] == 'IDLE':
        request_queue.put(['launch', query, 'test1'])
    else:
        request_queue.put(['proceed', query])
    while True:
        cmd, *args = response_queue.get()
        if cmd == 'new_msg':
            if history[-1][-1] is not None:
                history.append([None, None])
        elif cmd == 'delta':
            if history[-1][-1] is None:
                history[-1][-1] = ''
            history[-1][-1] += args[0]
            yield history, ""
        elif cmd == 'end':
            break
    return history, ""

with gr.Blocks() as demo:
    gr.Markdown(WEBUI_TITLE)
    with gr.Tab("对话"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot(elem_id="chat-box", show_label=False).style(height=600)
                query = gr.Textbox(show_label=False, placeholder="请输入提问内容，按回车进行提交").style(container=False)
            with gr.Column(scale=5):
                mode = gr.Radio(["LLM 对话"],
                                label="请选择使用模式",
                                value="LLM 对话")
            query.submit(get_answer,
                        [query, chatbot, mode],
                        [chatbot, query])


demo.queue(concurrency_count=3)

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0',
            server_port=17861,
            share=False,
            inbrowser=False)

import os
import sys
import queue
import threading
import gradio as gr
from uuid import uuid4
from pathlib import Path
from typing import List, Dict, Tuple, Type, Optional, Callable

import gradio as gr
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools
from langchain.agents import AgentExecutor
from langchain.requests import Requests
from langchain.tools.base import BaseTool
from langchain.chains.base import Chain
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
from utils.fetch_login_token import fetch_login_token

sys.path.append(str(Path(__file__).parent.joinpath("opsgpt-prompt")))

from opsgpt import (
    MetacubeToolkit,
    SearchToolkit,
    AlertseerToolkit,
    RiskseerToolkit,
    DataseerToolkit,    
    TicketseerToolkit,
    BizseerToolkit,
    OpsGPTAgent
) 

WEBUI_TITLE = '# Bizseer BMBOps'

CUSTOMIZED_CSS = """
.gradio-container-3-34-0 .scroll-hide::-webkit-scrollbar {
  display: block;
  width: 5px;
  height: 10px;
}
.gradio-container-3-34-0 .scroll-hide::-webkit-scrollbar-thumb {
  background: lightgray;
}
"""

BUILTIN_MODES = [
    # "direct_chat",
    "human_as_tool",
]

BIZSEER_TOOLKITS = [
    "metacube",
    "ticketseer",
    "alertseer",
    "riskseer",
    "dataseer",
]

BIZSEER_TOOLKITS_MAP: Dict[str, Tuple[Type[BizseerToolkit]]] = {
    name: clzs
    for name, clzs in zip(
        BIZSEER_TOOLKITS,
        [
            (MetacubeToolkit, SearchToolkit),
            (TicketseerToolkit, ),
            (AlertseerToolkit, ),
            (RiskseerToolkit, ),
            (DataseerToolkit, ),
            (TicketseerToolkit, )
        ]
    )
}

AVAILABLE_MODES = BIZSEER_TOOLKITS

request_queue = queue.Queue()
response_queue = queue.Queue()

load_dotenv(Path(__file__).parent.joinpath(".env"))

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
    agent: Chain
    reply: queue.Queue
    msg: str

    def __init__(self, agent: Chain, reply: queue.Queue, msg: str):
        super().__init__()
        self.agent = agent
        self.reply = reply
        self.msg = msg

    def run(self):
        self.agent.run(self.msg)
        self.reply.put(['end'])
        
class TaskController:
    reply: queue.Queue()
    
    _mode: str
    _agent: Optional[AgentExecutor]

    _task: Optional[Task]
    _msgbuf: Optional[queue.Queue]

    def __init__(self):
        super().__init__()
        self.reply = queue.Queue()
        
        self._mode = 'unknown'
        self._agent = None

        self._task = None
        self._msgbuf = None

    def ensure_mode(self, name):
        if self._mode == name:
            return 
        
        self._mode = name

        callback = CustomizedCallbackHandler(self.reply)

        self._msgbuf = queue.Queue()
        def input_func():
            self.reply.put(['end'])
            return self._msgbuf.get()
        
        self._agent = create_bizseer_agents(name, callback, input_func)

    def launch(self, msg: str):
        self._task = Task(self._agent, self.reply, msg)
        self._task.start()

    def proceed(self, msg: str):
        if self._msgbuf is not None:
            self._msgbuf.put(msg)

    def is_idle(self) -> bool:
        return self._task is None or not self._task.is_alive()

task_controller_map: Dict[str, TaskController] = {}

def get_task_controller(session_id: str) -> TaskController:
    if task_controller_map.get(session_id) is None:
        task_controller_map[session_id] = TaskController()    
    return task_controller_map[session_id]

def get_answer(query: str, history: List[List[str]], mode: str, session_id: str):
    history.append([query, None])
    yield history, ""
    taskctl = get_task_controller(session_id)
    taskctl.ensure_mode(mode)
    if taskctl.is_idle():
        taskctl.launch(query)
    else:
        taskctl.proceed(query)

    while True:
        cmd, *args = taskctl.reply.get()
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


def create_human_as_tools(callback: BaseCallbackHandler, input_func: Callable) -> List[BaseTool]:
    llm = ChatOpenAI(model_name='gpt-4-0613', streaming=True, callbacks=[callback], temperature=0)
    return load_tools(
        ["human"], 
        llm=llm,
        input_func=input_func,
    )


def create_bizseer_agents(toolkit_name: str, callback: BaseCallbackHandler, input_func: Callable) -> Chain:
    llm = OpenAI(model_name="text-davinci-003", streaming=True, callbacks=[callback], max_tokens=-1)

    auth_token = fetch_login_token()
    requests = Requests(headers={"Authorization": auth_token})

    clzs = BIZSEER_TOOLKITS_MAP[toolkit_name]
    tools = []
    for clz in clzs:
        tools.extend(
            clz.from_llm(llm, requests=requests).get_tools()
        )
    
    tools.extend(create_human_as_tools(callback=callback, input_func=input_func))
    
    return OpsGPTAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
        verbose=True
    )

def load_agent(session_id: str, mode: str):
    taskctl = get_task_controller(session_id)
    taskctl.ensure_mode(mode)
    
    def simple_name(name):
        return name.split(".")[-1].strip()
    
    return [
        [simple_name(tool.name), tool.description.strip()] for tool in taskctl._agent.tools
    ]


with gr.Blocks(css=CUSTOMIZED_CSS) as demo:
    gr.Markdown(WEBUI_TITLE)

    session_id = gr.State(lambda: str(uuid4()))

    with gr.Tab("对话"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot(elem_id="chat-box", show_label=False)
                chatbot.style(height=600)
                query = gr.Textbox(show_label=False, placeholder="请输入提问内容，按回车进行提交")
                query.style(container=False)
                table = gr.Dataframe(
                    headers=["工具", "描述"],
                    label="当前Agent可能使用的工具",
                    interactive=False,
                    wrap=True
                )
            with gr.Column(scale=5):
                mode = gr.Dropdown(AVAILABLE_MODES,
                                   label="请选择工具库",
                                   value=AVAILABLE_MODES[0],
                                   multiselect=False)
                clear_btn = gr.Button(value="清空对话历史")

            query.submit(get_answer,
                        [query, chatbot, mode, session_id],
                        [chatbot, query])
            
            clear_btn.click(lambda: [], [], [chatbot])
            mode.change(load_agent, inputs=[session_id, mode], outputs=[table])

    demo.load(load_agent, inputs=[session_id, mode], outputs=[table])
demo.queue(concurrency_count=3)

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0',
                server_port=9860,
                share=False,
                inbrowser=False)

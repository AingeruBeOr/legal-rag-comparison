import gradio as gr
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.online_pipeline import OnlinePipeline
from dotenv import load_dotenv
load_dotenv(".env")

pipeline = OnlinePipeline()

def respond(message, chat_history):
    text_response = pipeline.do_rag(message)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": text_response})
    return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()
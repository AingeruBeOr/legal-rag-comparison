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
    text_response, retrieved_context = pipeline.do_rag(message)

    response_to_render = f"Answer: {text_response}\n\nRetrieved Contexts:\n"
    for idx, context in enumerate(retrieved_context):
        response_to_render += f"{idx+1}. (Score: {context.score:.2f}) {context.payload['source_file']}\n\n"

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": text_response})
    return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()
import gradio as gr
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.online_pipeline import OnlinePipeline
from dotenv import load_dotenv
load_dotenv(".env")

pipeline = OnlinePipeline(
    embedding_model_name="BAAI/bge-m3",
    top_k=3, 
    collection_name="documents",
    llm_model_name="gemini-3.1-flash-lite"
)

def respond(message, chat_history):
    text_response, retrieved_context = pipeline.do_rag(message)

    response_to_render = f"# Respuesta\n{text_response}\n\n# Contexto recuperado\n"
    for idx, context in enumerate(retrieved_context):
        response_to_render += f"{idx+1}. (Score: {context.score:.2f}) {context.payload['source_file']} (page {context.payload['page']})\n\n"

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response_to_render})
    return "", chat_history

with gr.Blocks(fill_height=True) as demo:
    chatbot = gr.Chatbot(scale=2)
    msg = gr.Textbox(submit_btn=True, scale=1, show_label=False, placeholder="Escribe tu pregunta aquí...")
    clear = gr.ClearButton([msg, chatbot], value="Limpiar chat")

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()
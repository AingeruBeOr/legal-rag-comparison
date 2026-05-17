import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv("../.env")
from evaluation.llm_judge_inferencer import LLMJudgeInferencer

inferencer = LLMJudgeInferencer()

response = inferencer.judge("¿Cuál es la capital de Francia? (Devuelve la respuesta en formato JSON)")
print(response)
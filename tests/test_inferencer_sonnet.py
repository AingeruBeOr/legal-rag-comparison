import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inferencer_bedrock import SonnetInferencer
from dotenv import load_dotenv
load_dotenv("../.env")

inferencer = SonnetInferencer()

response = inferencer.infer("What is the capital of France?")
print(response)
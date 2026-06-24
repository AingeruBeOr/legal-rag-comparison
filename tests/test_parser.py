import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.parser import PDFParser

test_pdf_path = "../data/raw/BOE-A-2024-2248.pdf"
parser = PDFParser(test_pdf_path)

documents = parser.parse()

print("Número de documentos", len(documents))
print(documents[0])
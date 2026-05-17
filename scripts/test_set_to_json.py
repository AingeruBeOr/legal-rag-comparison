import pandas as pd
import json
import os

def csv_to_rag_json(csv_path: str, json_path: str):
    # Leer el archivo CSV
    df = pd.read_csv(csv_path)
    
    # Rellenar hacia adelante (forward fill) los valores vacíos en 'doc' y 'pág(s)'
    # Esto ocurre porque en el CSV a veces vienen celdas combinadas o implícitas
    df['doc'] = df['doc'].ffill()
    df['pág(s)'] = df['pág(s)'].ffill()
    
    # Crear la lista de diccionarios con la estructura para evaluación de RAG (Ragas / TruLens / etc)
    json_data = []
    for _, row in df.iterrows():
        if pd.isna(row['Pregunta']) or pd.isna(row['Respuesta']):
            continue
            
        item = {
            "question": str(row['Pregunta']).strip(),
            "target": str(row['Respuesta']).strip(),
            "metadata": {
                "source": str(row['doc']).strip(),
                "pages": str(row['pág(s)']).strip()
            }
        }
        json_data.append(item)
    
    # Guardar a JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
        
    print(f"Convertidos {len(json_data)} registros y guardados en {json_path}")

if __name__ == "__main__":
    # Ajustar rutas asumiendo que el script está en scripts/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = os.path.join(base_dir, "data", "test_set.csv")
    json_file = os.path.join(base_dir, "data", "test_set.json")
    
    csv_to_rag_json(csv_file, json_file)

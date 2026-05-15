# Comparativa de soluciones basadas en sistemas RAG sobre documentación legal

## Data

Dataset availability: TBA (Hugginface Datasets)

(https://alumnosunir-my.sharepoint.com/my?id=%2Fpersonal%2Faingeru%5Fbellido984%5Fcomunidadunir%5Fnet%2FDocuments%2FTFM%2FDataset&viewid=58c26454%2Dedbb%2D4630%2Dbdd5%2D2c69555beb97)

## Develop

`pip install -r requirements.txt`

## Components

### Vector DB

Using Qdrant. Options:
- in-memory
- on disk
- Docker: 
  - start: `docker run -p 6333:6333 qdrant/qdrant`
  - UI: http://localhost:6333/dashboard


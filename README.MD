# RAG Evaluation Framework

Proyecto para comparar configuraciones de un sistema RAG sobre un corpus propio de PDFs. El flujo permite:

- leer documentos colocados manualmente en `data/corpus`;
- probar distintas estrategias de chunking y retrieval definidas en `config/*.json`;
- generar respuestas sobre un golden dataset local;
- evaluar los resultados con RAGAS;
- visualizar comparativas en un dashboard de Streamlit.

## Estructura

```text
.
|-- config/                    # Configuraciones de experimentos que si se versionan
|-- data/
|   |-- corpus/                # Coloca aqui tus PDFs. Su contenido no se sube a git
|   `-- golden_dataset/        # Coloca aqui tu golden dataset. Su contenido no se sube a git
|-- results/
|   |-- intermediate_files/    # Salidas de main.py antes de evaluar. No se suben a git
|   `-- final_results/         # Resultados finales de RAGAS. No se suben a git
|-- src/
|   |-- chunking.py            # Estrategias de chunking
|   `-- retriever.py           # Estrategias de retrieval
|-- main.py                    # Indexa, recupera y genera respuestas
|-- evaluation_rag.py          # Evalua las respuestas con RAGAS
|-- dashboard.py               # Dashboard para comparar resultados
|-- requirements_rag.txt       # Dependencias del pipeline RAG
`-- requirements_eval.txt      # Dependencias de evaluacion y dashboard
```

Las carpetas de datos y resultados se mantienen en git mediante archivos `.gitkeep`, pero sus contenidos reales quedan ignorados.

## Preparacion

1. Crea y activa un entorno virtual.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instala dependencias.

```powershell
pip install -r requirements_rag.txt
pip install -r requirements_eval.txt
```

3. Crea un archivo `.env` local con tus claves. No se sube a git.

```env
OPENAI_API_KEY=tu_clave
AZURE_SEARCH_ENDPOINT=https://tu-servicio.search.windows.net
AZURE_SEARCH_KEY=tu_clave_de_azure_search
AZURE_SEARCH_CLIENT=nombre_del_indice
```

## Datos de entrada

Coloca tus documentos PDF en:

```text
data/corpus/
```

Coloca tu golden dataset en:

```text
data/golden_dataset/golden_dataset_single-turn.jsonl
```

Cada linea del golden dataset debe ser un JSON con esta estructura minima:

```json
{
  "user_input": "Pregunta que se hara al RAG",
  "reference_contexts": ["Contexto esperado o fragmentos de referencia"],
  "reference": "Respuesta esperada",
  "metadata": {}
}
```

## Configuracion de experimentos

Los experimentos se definen en `config/*.json`. Cada archivo debe incluir:

- `experiment_name`: nombre usado para guardar los resultados;
- `chunking.strategy`: `recursive`, `semantic` o `hierarchical`;
- `chunking.params`: parametros de chunking;
- `retrieval.method`: `vector` o `hybrid`;
- `retrieval.params`: parametros como `top_k`, `reranking` y `top_n`.

Ejemplo simplificado:

```json
{
  "experiment_name": "semantic_vector_k5",
  "chunking": {
    "strategy": "semantic",
    "params": {
      "chunk_size": 100,
      "embedding": "openai://text-embedding-3-small",
      "threshold": 0.8,
      "similarity_window": 3,
      "skip_window": 1
    }
  },
  "retrieval": {
    "method": "vector",
    "params": {
      "reranking": false,
      "top_k": 5
    }
  }
}
```

## Ejecucion

1. Genera respuestas para los experimentos pendientes.

```powershell
python main.py
```

Este paso lee los PDFs de `data/corpus`, crea el indice en Azure AI Search, ejecuta las preguntas del golden dataset y guarda salidas en `results/intermediate_files`.

2. Evalua con RAGAS.

```powershell
python evaluation_rag.py
```

Este paso procesa los JSON intermedios y guarda resultados finales en `results/final_results`.

3. Abre el dashboard.

```powershell
streamlit run dashboard.py
```

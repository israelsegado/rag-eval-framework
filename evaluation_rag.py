import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from ragas.llms import llm_factory
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
#from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential
from ragas.run_config import RunConfig

#-------------------------------------- CARGA DE CLAVES -------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ------------- Declaración de los modelos (unificamos en un sitio para posibles cambios)
EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
GEN_MODEL = "gpt-5.4-mini"
#GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
#===================================================================================================

#------------------------------------------ EVALUACIÓN RAGAS ---------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_FILEPATH = CURRENT_DIR / "results" / "final_results"
INTERMEDIATE_FILEPATH =  CURRENT_DIR / "results" / "intermediate_files"

llm = ChatOpenAI(
    model=GEN_MODEL,
    temperature=0.0,
    max_completion_tokens=2048,
    max_retries=3
)

JUDGE_LLM = LangchainLLMWrapper(llm)
JUDGE_EMB = LangchainEmbeddingsWrapper(EMBEDDING_MODEL)

processed = {f.name for f in RESULTS_FILEPATH.glob("*.json")}

for process_file in INTERMEDIATE_FILEPATH.glob("*.json"):
    if process_file.name in processed:
        print(f"El achivo {process_file.name} ya ha sido procesado.")
        continue

    print(f"Procesando archivo: {process_file.name}...")
    with open(INTERMEDIATE_FILEPATH/ process_file, 'r', encoding="utf-8") as f:
        to_process_file = json.load(f)
            
    samples = []
    for row in to_process_file["results"]:
        sample = SingleTurnSample(
            user_input=row["user_input"],
            response=row["response"],
            retrieved_contexts=row["retrieved_contexts"],
            reference_contexts=row["reference_contexts"],
            reference=row["reference"]
        )
        samples.append(sample)

    print("Evaluando con RAGAS...")
    dataset = EvaluationDataset(samples=samples)

    try:
        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=2, min=10, max=180),
            reraise=True
        )
        def safe_ragas_evaluate(dataset):
            return evaluate(
                dataset=dataset,
                metrics=[Faithfulness(llm=JUDGE_LLM),  
                        AnswerRelevancy(llm=JUDGE_LLM, embeddings=JUDGE_EMB), 
                        #ContextRecall(llm=JUDGE_LLM), 
                        #ContextPrecision(llm=JUDGE_LLM)],
                        ],
                run_config=RunConfig(max_workers=1, max_retries=3, timeout=120, max_wait=180)
            )
        
        result_ragas = safe_ragas_evaluate(dataset)

        ragas_df = result_ragas.to_pandas()

        #ragas_df["context_precision"] = ragas_df["context_precision"].fillna(0.0)
        ragas_df["faithfulness"] = ragas_df["faithfulness"].fillna(0.0)
        ragas_df["answer_relevancy"] = ragas_df["answer_relevancy"].fillna(0.0)
        #ragas_df["context_recall"] = ragas_df["context_recall"].fillna(0.0)

        detail_records = ragas_df.to_dict(orient="records")
        for detail_record, original_row in zip(detail_records, to_process_file["results"]):
            detail_record["retrieved_metadata"] = original_row.get("retrieved_metadata", [])
            detail_record["metadata"] = original_row.get("metadata", {})

        output = to_process_file.copy()
        output["results"] = {
            "Faithfulness": ragas_df["faithfulness"].mean(),
            "Answer_Relevancy": ragas_df["answer_relevancy"].mean(),
            #"Context_Recall": ragas_df["context_recall"].mean(),
            #"Context_Precision": ragas_df["context_precision"].mean(),
            "detail": detail_records
        }

        output_path = RESULTS_FILEPATH / f"{to_process_file['data_config']['experiment_name']}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print("Evaluación terminada con éxito.")
        print(f"Resultados de {process_file.name} guardados en {RESULTS_FILEPATH}.")
    except:
        print(f" XXX El achivo {process_file.name} no ha podido ser evaluado. XXX")
        import traceback
        traceback.print_exc()
        print("Continuando con el siguiente...")
#==================================================================================

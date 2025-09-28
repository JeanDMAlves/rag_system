#%%
import os
import json
from app.utils.rag import RAG
from app.utils.db import get_embeddings, save_interaction
from dotenv import load_dotenv

load_dotenv('./.env')
filename = 'manual-siafi.pdf'

rag_system = RAG(
    chunk_size = 400, 
    chunk_overlap = 20, 
    top_k_retrieval = 80,
    top_k_rerank = 16,
    cross_encoder_threshold = 0.2,
    embedding_model = os.environ.get('EMBEDDING_MODEL'),
    cross_encoder_model = os.environ.get('CROSS_ENCODER_MODEL'),
    llm_generation_model = os.environ.get('LLM_GENERATION_MODEL'),
    openrouter_api_key = os.environ.get('OPENROUTER_API_KEY')
) 

chunks, embeddings = get_embeddings(filename)
embedding_model = rag_system.load_embedding_model()
cross_encoder_model = rag_system.load_cross_encoder()

with open('./experimentos/ground_truth.json', 'r', encoding='utf-8') as f:
    ground_truth = json.load(f)

# %%
from experimentos.metrics import *
from tqdm import tqdm

results = []
for entry in tqdm(ground_truth, desc='Avaliando perguntas'):
    question = entry['question']
    relevant_chunks = entry['relevant_chunks']
    
    # Rodar a resposta
    rag_result = rag_system.full_rag_system(
        chunks=chunks,
        passage_embeddings=embeddings,
        embedding_model=embedding_model,
        cross_encoder_model=cross_encoder_model,
        question=question
    )

    resposta = rag_result['resposta']
    pred_chunks = [chunks[idx] for idx in rag_result['chunks_utilizados']]

    results.append({
        'pergunta': question,
        'resposta': resposta,
        'pred_chunks': pred_chunks
    })

#%%
df = evaluate_with_bertscore(results, ground_truth[0:16])
print(df)
df.to_csv("./experimentos/bertscore_results.csv", index=False)
# %%
df.columns
# %%
for _, row in df.iterrows():
    print(f'PERGUNTA: {row['pergunta']}')
    print(row['resposta_rag'])
# %%
df[['bertscore_precision', 'bertscore_recall', 'bertscore_f1']]
# %%

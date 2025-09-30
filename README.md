# rag_system

Este projeto implementa um sistema **RAG (Retrieval-Augmented Generation)** para responder perguntas com base em documentos.  
Além disso, inclui métricas automáticas de avaliação (ex.: **BERTScore**) para comparar as respostas geradas com um **ground truth** previamente anotado.
Os resultados estão na pasta /experimentos/. Os arquivos referentes ao sistema desenvolvido usando Streamlit estão dentro da pasta /app/

## 🚀 Tecnologias

- [uv](https://github.com/astral-sh/uv) – Gerenciamento de ambientes Python
- [Transformers](https://huggingface.co/docs/transformers) – Modelos de embeddings e LLMs
- [Sentence-Transformers](https://www.sbert.net/) – Similaridade semântica e recuperação
- [BERTScore](https://github.com/Tiiiger/bert_score) – Avaliação das respostas
- [Pandas](https://pandas.pydata.org/) – Manipulação de dados e exportação de resultados
- [OpenRouter API](https://openrouter.ai/) – Acesso a LLMs gratuitos/externos

---

## 📦 Instalação

Instalação do UV:
[https://docs.astral.sh/uv/getting-started/installation/]

Clone o repositório e crie o ambiente com **uv**, adicione um arquivo .env na raiz:

Exemplo de .env na raiz do projeto:

```bash
OPENROUTER_API_KEY=~~OpenRouterKey~~
EMBEDDING_MODEL=intfloat/e5-large-v2
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
LLM_GENERATION_MODEL=deepseek/deepseek-chat-v3.1:free
TOKENIZERS_PARALLELISM=false
```

Virtualenv usando UV:

```bash
# Criar e ativar ambiente virtual com uv
uv venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows PowerShell

# Instalar dependências
uv pip install -r requirements.txt
```

Se quiser sincronizar usando o UV:

```bash
uv sync
```



Iniciar o sistema RAG:

```bash
uv run streamlit run app/main.py
```

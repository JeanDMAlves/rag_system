"""
Aplicação RAG interativa com Streamlit
--------------------------------------

Este módulo implementa uma interface Streamlit para um sistema RAG (Retrieval-Augmented Generation).
O sistema permite que usuários carreguem documentos em formato `.txt` ou `.pdf`, 
realizem perguntas sobre o conteúdo e obtenham respostas geradas por um modelo de linguagem 
com suporte de busca híbrida (embeddings + cross-encoder).

Funcionalidades principais:
- Upload de documentos.
- Processamento e chunking do conteúdo.
- Geração e persistência de embeddings em banco de dados.
- Consulta de respostas anteriores armazenadas.
- Interação com o sistema RAG para geração de novas respostas.
- Histórico de interações por documento.
"""

import os
import streamlit as st
from utils.rag import RAG
from dotenv import load_dotenv
from components.history_iteraction import plot_iteraction
from utils.db import (
    create_db, 
    save_interaction, 
    add_embeddings, 
    get_embeddings, 
    list_interactions_by_filename, 
    search_answer_by_filename
)


def reset_system():
    """
    Função auxiliar para resetar o estado da aplicação.
    Remove todas as variáveis armazenadas em `st.session_state`.
    """
    for key in st.session_state.keys():
        del st.session_state[key]


@st.cache_resource
def get_rag_system():
    """
    Inicializa e retorna uma instância do sistema RAG.

    - Carrega variáveis de ambiente a partir do arquivo `.env`.
    - Cria o banco de dados (se não existir).
    - Configura o sistema RAG com os parâmetros definidos.

    Returns:
        RAG: instância configurada do sistema RAG.
    """
    load_dotenv(dotenv_path='./.env')
    create_db()
    rag_system = RAG(
        chunk_size=400, 
        chunk_overlap=20, 
        top_k_retrieval=80,
        top_k_rerank=16,
        cross_encoder_threshold=0.2,
        embedding_model=os.environ.get('EMBEDDING_MODEL'),
        cross_encoder_model=os.environ.get('CROSS_ENCODER_MODEL'),
        llm_generation_model=os.environ.get('LLM_GENERATION_MODEL'),
        openrouter_api_key=os.environ.get('OPENROUTER_API_KEY')
    )
    return rag_system


# Inicializa o sistema RAG
rag = get_rag_system()

# Define estado inicial
if 'chunks' not in st.session_state:
    st.session_state.chunks = None

# Configuração da página Streamlit
st.set_page_config(layout='wide', page_title='RAG System')

with st.container():
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("## Faça Perguntas sobre seus documentos!")
    with col2:
        # Botão para resetar documentos carregados
        st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
        st.button(
            'Resetar documentos', 
            on_click=reset_system,
            disabled=st.session_state.chunks is None,
            icon=':material/close:',
            type='primary'
        )
        st.markdown("</div>", unsafe_allow_html=True)

# Caso nenhum documento tenha sido carregado
if st.session_state.chunks is None:
    st.sidebar.write('## Faça Upload do arquivo aqui!')

    document_file = st.sidebar.file_uploader('Arquivo', type=['txt', 'pdf'])

    if document_file is not None:
        # Exibe informações básicas do arquivo
        st.sidebar.write("Nome:", document_file.name)
        st.sidebar.write("Tipo:", document_file.type)
        st.sidebar.write("Tamanho:", document_file.size, "bytes")

        with st.spinner('Processando o arquivo, aguarde...'):
            chunks, embeddings = get_embeddings(document_file.name)

            # Caso seja a primeira vez que o arquivo está sendo carregado
            if (chunks is None) and (embeddings is None):
                if document_file.type == 'text/plain':
                    content = rag.load_txt(document_file)
                elif document_file.type == 'application/pdf':
                    content = rag.load_pdf(document_file)
                else: 
                    content = 'Formato desconhecido'

                # Armazena variáveis na sessão
                st.session_state.document_file = document_file
                st.session_state.chunks = rag.chunking(content)
                st.session_state.embedding_model = rag.load_embedding_model()
                st.session_state.passage_embeddings = rag.embeddings(
                    embedding_model=st.session_state.embedding_model,
                    chunks=st.session_state.chunks
                )
                st.session_state.cross_encoder_model = rag.load_cross_encoder()

                # Persiste embeddings no banco
                add_embeddings(
                    filename=document_file.name,
                    chunks=st.session_state.chunks,
                    embeddings=st.session_state.passage_embeddings
                )
            else:
                # Carrega informações já armazenadas no banco
                st.session_state.document_file = document_file
                st.session_state.chunks = chunks
                st.session_state.embedding_model = rag.load_embedding_model()
                st.session_state.passage_embeddings = embeddings
                st.session_state.cross_encoder_model = rag.load_cross_encoder()

        # Recarrega a aplicação após upload
        st.rerun()

else:
    # Caso já exista documento carregado
    st.write(
        f'Quantidade de Chunks: {len(st.session_state.chunks)} '
        f'|| Dimensões dos embeddings: ({st.session_state.passage_embeddings.shape[0]}, '
        f'{st.session_state.passage_embeddings.shape[1]})'
    )

    # Área de input para perguntas do usuário
    question = st.text_area('Qual sua pergunta?')

    if st.button('Buscar resposta', icon=':material/search:'):
        if question.strip():
            with st.spinner('Buscando resposta...'):
                # Verifica se já existe resposta salva no banco
                answer_in_db = search_answer_by_filename(
                    document=st.session_state.document_file.name,
                    question=question
                )
                if answer_in_db:
                    st.write('Resposta:')
                    st.write(answer_in_db[0][0])
                else:
                    # Executa pipeline completo do RAG
                    answer = rag.full_rag_system(
                        chunks=st.session_state.chunks,
                        passage_embeddings=st.session_state.passage_embeddings,
                        embedding_model=st.session_state.embedding_model,
                        cross_encoder_model=st.session_state.cross_encoder_model,
                        question=question,
                    )
                    # Salva interação no banco
                    save_interaction(
                        document=st.session_state.document_file.name,
                        answer=answer['resposta'],
                        question=question,
                    )
        else:
            st.warning('Digite uma pergunta antes de buscar.')

    # Histórico de perguntas e respostas anteriores
    iteractions = list_interactions_by_filename(st.session_state.document_file.name)

    with st.container(border=True):
        st.write('### Histórico de Perguntas:')
        for iteraction in iteractions:
            plot_iteraction(st, iteraction, st.session_state.document_file.name)

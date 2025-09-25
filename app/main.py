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
    for key in st.session_state.keys():
        del st.session_state[key]

@st.cache_resource
def get_rag_system():
    load_dotenv(dotenv_path='./.env')
    create_db()
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
    return rag_system

rag = get_rag_system()

if 'chunks' not in st.session_state:
    st.session_state.chunks = None

st.set_page_config(layout='wide', page_title='RAG System')
with st.container():
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("## Faça Perguntas sobre seus documentos!")
    with col2:
        # Parte para resetar o documento do sistema
        # Apagando o documento enviado anteriormente 
        # e deixando o usuário enviar outro
        st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
        st.button(
            'Resetar documentos', 
            on_click = reset_system,
            disabled = st.session_state.chunks is None,
            icon=':material/close:',
            type='primary'
        )
        st.markdown("</div>", unsafe_allow_html=True)

# Se o documento não estiver carregado, apresentamos ao usuário a tela para carregar documento
if st.session_state.chunks is None:
    st.sidebar.write('## Faça Upload do arquivo aqui!')

    col1, col2 = st.columns(2)
    document_file = st.sidebar.file_uploader('Arquivo', type=['txt', 'pdf'])

    if document_file is not None:
        st.sidebar.write("Nome:", document_file.name)
        st.sidebar.write("Tipo:", document_file.type)
        st.sidebar.write("Tamanho:", document_file.size, "bytes")

        with st.spinner('Processando o arquivo, aguarde...'):
            chunks, embeddings = get_embeddings(document_file.name)

            # Carrega pela primeira vez
            if (chunks is None) and (embeddings is None):
                # Carrega as linhas do documento
                if document_file.type == 'text/plain':
                    content = rag.load_txt(document_file)
                elif document_file.type == 'application/pdf':
                    content = rag.load_pdf(document_file)
                else: 
                    content = 'Formato desconhecido'
                st.session_state.document_file = document_file
                st.session_state.chunks = rag.chunking(content)
                st.session_state.embedding_model = rag.load_embedding_model()
                st.session_state.passage_embeddings = rag.embeddings(
                    embedding_model=st.session_state.embedding_model,
                    chunks=st.session_state.chunks
                )
                st.session_state.cross_encoder_model = rag.load_cross_encoder()
                add_embeddings(
                    filename=document_file.name,
                    chunks=st.session_state.chunks,
                    embeddings=st.session_state.passage_embeddings
                )
            else:
                # Carrega do banco de dados se o documento já tiver sido carregado alguma outra vez
                # Não precisamos de file_content pq já temos os chunks e embeddings calculados
                st.session_state.document_file = document_file
                st.session_state.chunks = chunks
                st.session_state.embedding_model = rag.load_embedding_model()
                st.session_state.passage_embeddings = embeddings
                st.session_state.cross_encoder_model = rag.load_cross_encoder()

        st.rerun()
else: 
    # Se o documento já estiver carregado (!= None), carregamos a página para fazer perguntas
    st.write(f'Quantidade de Chunks: {len(st.session_state.chunks)} || Dimensões dos embeddings: ({st.session_state.passage_embeddings.shape[0]}, {st.session_state.passage_embeddings.shape[1]})')
    question = st.text_area('Qual sua pergunta?')

    if st.button(
        'Buscar resposta',
        icon=':material/search:',
        ):
        if question.strip():
            with st.spinner('Buscando resposta...'):
                answer_in_db = search_answer_by_filename(
                    document=st.session_state.document_file.name,
                    question=question
                )
                if answer_in_db:
                    st.write('Resposta:')
                    st.write(answer_in_db[0][0])
                else: 
                    answer = rag.full_rag_system(
                        chunks=st.session_state.chunks,
                        passage_embeddings=st.session_state.passage_embeddings,
                        embedding_model=st.session_state.embedding_model,
                        cross_encoder_model=st.session_state.cross_encoder_model,
                        question=question,
                    )
                    save_interaction(
                        document=st.session_state.document_file.name,
                        answer=answer['resposta'],
                        question=question,
                    )
                    st.write('Resposta:')
                    st.write(answer['resposta'])
        else:
            st.warning('Digite uma pergunta antes de buscar.')
    
    iteractions = list_interactions_by_filename(st.session_state.document_file.name)
    
    with st.container(
        border=True
    ):
        st.write('### Histórico de Perguntas:')
        for iteraction in iteractions:
            plot_iteraction(st, iteraction)



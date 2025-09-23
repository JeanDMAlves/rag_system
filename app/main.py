import streamlit as st
from utils.rag import RAG
from dotenv import load_dotenv
import os 

@st.cache_resource
def get_rag_system():
    load_dotenv(dotenv_path='./.env')
    rag_system = RAG(
        chunk_size = 400, 
        chunk_overlap = 20, 
        top_k_retrieval = 80,
        top_k_rerank = 16,
        cross_encoder_threshold = 0.2,
        embedding_model = os.environ.get('EMBEDDING_MODEL'),
        cross_encoder_model = os.environ.get('CROSS_ENCODER_MODEL'),
        llm_generation_model = os.environ.get('LLM_GENERATION_MODEL')
    )
    return rag_system
rag = get_rag_system()

if 'file_content' not in st.session_state:
    st.session_state.file_content = None

st.set_page_config(layout='wide', page_title='RAG System')
st.write('## Faça Perguntas sobre seus documentos!')

if st.session_state.file_content is None:
    st.sidebar.write('## Faça Upload do arquivo aqui!')

    col1, col2 = st.columns(2)
    document_file = st.sidebar.file_uploader('Arquivo', type=['txt', 'pdf'])

    if document_file is not None:
        st.sidebar.write("Nome:", document_file.name)
        st.sidebar.write("Tipo:", document_file.type)
        st.sidebar.write("Tamanho:", document_file.size, "bytes")

        with st.spinner('Processando o arquivo, aguarde...'):
            if document_file.type == 'text/plain':
                content = rag.load_txt(document_file)
            elif document_file.type == 'application/pdf':
                content = rag.load_pdf(document_file)
            else: 
                content = 'Formato desconhecido'

            st.session_state.file_content = content
            st.session_state.chunks = rag.chunking(st.session_state.file_content)
            st.session_state.embedding_model = rag.load_embedding_model()
            st.session_state.passage_embeddings = rag.embeddings(
                embedding_model=st.session_state.embedding_model,
                chunks=st.session_state.chunks
            )
            st.session_state.cross_encoder_model = rag.load_cross_encoder()
        
        st.success('Arquivo carregado com sucesso!')
if st.session_state.file_content is not None:
    
    question = st.text_area('Qual sua pergunta sobre o documento?')
    
    if question:
        answer = rag.full_rag_system(
            chunks=st.session_state.chunks,
            passage_embeddings=st.session_state.passage_embeddings,
            embedding_model=st.session_state.embedding_model,
            cross_encoder_model=st.session_state.cross_encoder_model,
            question=question,
        )
        st.write(answer)
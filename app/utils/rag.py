import pymupdf
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple
from torch.nn.functional import cosine_similarity
import torch
import requests
import time
import os
import math

class RAG:
    def __init__(
            self, 
            chunk_size = 400, 
            chunk_overlap = 20, 
            top_k_retrieval = 80,
            top_k_rerank = 16,
            cross_encoder_threshold = 0.2,
            embedding_model = 'intfloat/e5-large-v2',
            cross_encoder_model = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            llm_generation_model = 'openai/gpt-4o',
            openrouter_api_key = os.environ.get('OPENROUTER_API_KEY')
        ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.cross_encoder_threshold = cross_encoder_threshold
        self.embedding_model = embedding_model
        self.cross_encoder_model = cross_encoder_model
        self.llm_generation_model = llm_generation_model
        self.openrouter_api_key = openrouter_api_key

    # Verifica se existe GPU (CUDA) ou não
    def find_device(self) -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Ler o documento de texto (PDF)
    def load_pdf(self, path):
        doc = pymupdf.open(stream=path.read(), filetype='pdf')
        raw_text = ""
        for page in doc:
            raw_text += page.get_text("text")
        return raw_text
    
    # Ler o documento de texto (txt)
    def load_txt(self, file):
        file.seek(0)
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            file.seek(0)
            try:
                content = file.read().decode(encoding)
                file.seek(0)
                return content
            except UnicodeDecodeError:
                file.seek(0)
                continue
        raise Exception('Não foi possível ler o arquivo enviado!')
    
    # Dividir o documento em pedaços menores e gerenciáveis
    def chunking(self, text: str):
        tokenizer = AutoTokenizer.from_pretrained(self.embedding_model, use_fast=True)
        max_allowed = tokenizer.model_max_length if hasattr(tokenizer, "model_max_length") else 512

        safe_max = max_allowed - 32
        chunk_size_tokens = min(self.chunk_size, safe_max)

        # overlap em tokens
        tokens_overlap = int(chunk_size_tokens * (self.chunk_overlap / 100))
        step_tokens = max(1, chunk_size_tokens - tokens_overlap)

        # estima chars por token com uma amostra (mais preciso que heurística fixa)
        sample_text = text[:2000]  # 2000 chars é suficiente normalmente
        try:
            sample_tokens = tokenizer.encode(sample_text, add_special_tokens=False)
            if len(sample_tokens) > 0:
                avg_chars_per_token = len(sample_text) / len(sample_tokens)
            else:
                avg_chars_per_token = 4.0
        except Exception:
            avg_chars_per_token = 4.0

        # calcula tamanhos em caracteres para as fatias iniciais
        char_chunk = max(200, int(math.ceil(chunk_size_tokens * avg_chars_per_token)))
        char_step = max(50, int(math.ceil(step_tokens * avg_chars_per_token)))

        text_len = len(text)
        text_chunks = []
        
        # varre o texto em janelas de caracteres (não tokeniza tudo)
        for start_char in range(0, text_len, char_step):
            end_char = min(text_len, start_char + char_chunk)
            candidate = text[start_char:end_char]

            # tokeniza apenas a fatia
            token_ids = tokenizer.encode(candidate, add_special_tokens=False)

            if len(token_ids) <= chunk_size_tokens:
                # cabe num chunk só
                chunk_text = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                if chunk_text.strip():
                    text_chunks.append(chunk_text)
            else:
                # se a fatia ainda é maior que chunk_size em tokens,
                # divide a lista de ids em subchunks token-wise (com overlap)
                for i in range(0, len(token_ids), step_tokens):
                    sub_ids = token_ids[i:i + chunk_size_tokens]
                    if not sub_ids:
                        continue
                    chunk_text = tokenizer.decode(sub_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    if chunk_text.strip():
                        text_chunks.append(chunk_text)

            # Early Stopping
            if end_char == text_len:
                break
        
        # remover chunks muito curtos ou duplicados consecutivos
        cleaned = []
        prev = None
        for c in text_chunks:
            s = c.strip()
            if len(s) < 10:
                continue
            if prev is not None and s == prev:
                continue
            cleaned.append(s)
            prev = s

        return cleaned
    
    def chunking_rcts(self, text: str):
        """
        Divide o texto em pedaços coesos (sem quebrar palavras) usando RecursiveCharacterTextSplitter
        e garante que cada pedaço não ultrapasse o limite de tokens do modelo de embedding.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)

        # Split inicial com RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 4,   # usa caracteres como proxy (4 chars ≈ 1 token)
            chunk_overlap=int(self.chunk_size * self.chunk_overlap / 100) * 4,
            length_function=len,
        )
        initial_chunks = splitter.split_text(text)

        # Ajustar cada pedaço com base no tokenizer
        final_chunks = []
        for chunk in initial_chunks:
            tokens = tokenizer.encode(chunk, add_special_tokens=False)

            # Se o chunk couber no limite, mantém
            if len(tokens) <= self.chunk_size:
                final_chunks.append(chunk)
                continue

            # Se for maior que o limite, divide em sub-chunks
            step = self.chunk_size - int(self.chunk_size * self.chunk_overlap / 100)
            for i in range(0, len(tokens), step):
                sub_tokens = tokens[i:i + self.chunk_size]
                sub_chunk = tokenizer.decode(sub_tokens, skip_special_tokens=True)
                if sub_chunk.strip():
                    final_chunks.append(sub_chunk)

        return final_chunks
    
    # Carrega o modelo usado para Embedding no CUDA se possível
    def load_embedding_model(self) -> SentenceTransformer:
        embedding_model = SentenceTransformer(self.embedding_model)
        device = self.find_device()
        if device == 'cuda':
            embedding_model = embedding_model.to(device)
        return embedding_model
    
    # Carrega o modelo de Cross Encoder usado no Re-Ranking
    def load_cross_encoder(self) -> CrossEncoder:
        return CrossEncoder(self.cross_encoder_model)
    
    # Converte cada chunk em vetores numéricos que capturam o significado
    def embeddings(self, chunks, embedding_model: SentenceTransformer):
        passages_prefixed = [f"passage: {chunk}" for chunk in chunks]
        passage_embeddings = embedding_model.encode(
            passages_prefixed,
            convert_to_tensor = True,
            show_progress_bar = True,
            device = self.find_device(),
            batch_size = 64
        )
        return passage_embeddings

    # Faz a chamada à LLM disponibilizada pela OpenRouter
    def llm_api(self, prompt: str, system_prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openrouter_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.llm_generation_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 500
                    }    
                )
                response.raise_for_status()
                result = response.json()['choices'][0]['message']['content']
                return result.strip()
            except requests.exceptions.RequestException as e:
                print(f"Tentativa {attempt + 1} falhou: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Status: {e.response.status_code}")
                    print(f"Resposta: {e.response.text[:200]}")

                if attempt == max_retries - 1:
                    print(f"Todas as {max_retries} tentativas falharam")
                    return ""

                time.sleep(2 ** attempt)
        return ""

    # Expansão de consulta: Cria-se variações da pergunta para melhorar a busca
    # Expansão Multi Query
    def multi_query_extention(self, query: str):
        system_prompt = """
            Você é um assistente especializado em reformular perguntas.
            Sua tarefa é gerar 4 reformulações diferentes de uma pergunta para melhorar a busca em documentos.
            As reformulações devem:
            - Manter o mesmo significado da pergunta original
            - Usar sinônimos e estruturas diferentes
            - Ser claras e diretas
            - Cobrir diferentes formas de expressar a mesma ideia

            Retorne apenas as 4 perguntas, uma por linha, sem numeração ou marcadores.
        """

        response = self.llm_api(
            prompt = query,
            system_prompt = system_prompt,
            max_retries = 3
        )

        if response:
            # Extrair as perguntas da resposta
            queries = [q.strip() for q in response.split('\n') if q.strip()]
            queries = [q for q in queries if len(q) > 5] 
            return queries
        
        return []
    
    # Query Extention:
    # Hyphotetical Document Embeddings (HyDE)
    def hyde_query_extention(self, query: str):
        system_prompt = """
            Você é um assistente especializado.
            Gere uma resposta hipotética para a pergunta do usuário, como se fosse um trecho
            extraído de um documento.

            A resposta deve:
            - Ser plausível e bem estruturada
            - Ter entre 2-4 frases
            - Usar vocabulário típico de um especialista
            - Não afirmar certezas absolutas (use "parece", "sugere", "indica", etc.)

            Não mencione que é uma resposta hipotética.
        """

        response = self.llm_api(query, system_prompt, max_retries=3)
        return response if response else ""
        
    # Multi-query + HyDE
    # Encontramos os pedaços mais relevantes usando similaridade vetorial
    def hybrid_search(
            self, 
            chunks: List[str],
            query_original: str, 
            multi_queries: List[str], 
            hyde_doc: str, 
            passage_embeddings: torch.Tensor,
            embedding_model: SentenceTransformer,
            top_k: int = 80
        ) -> Tuple[List[str], List[int], List[float]]:
        all_queries = [query_original] + multi_queries
        if hyde_doc:
            all_queries.append(hyde_doc)
        
        queries_prefixed = [f'query: {q}' for q in all_queries]
        query_embeddings = embedding_model.encode(
            queries_prefixed,
            convert_to_tensor=True,
            show_progress_bar=True
        )

        query_embeddings = query_embeddings.to(passage_embeddings.device)
        # Calcular similaridade entre todas as consultas e todos os passages
        similarities = torch.mm(query_embeddings, passage_embeddings.T)

        # Para cada passage, pegar o score máximo entre todas as consultas
        max_scores, best_query_idx = torch.max(similarities, dim=0)
        
        # Pegar os top-k resultados
        top_scores, top_indices = torch.topk(max_scores, k=min(top_k, len(chunks)))

        # Converter para listas Python
        top_indices = top_indices.cpu().tolist()
        top_scores = top_scores.cpu().tolist()

        # Recuperar os chunks correspondentes
        retrieved_chunks = [chunks[i] for i in top_indices]

        return retrieved_chunks, top_indices, top_scores

    # Refinamos os resultados com um modelo mais preciso
    # Utilizando do Cross-Encoder
    # Recebe a pergunta + chunk juntos como entrada
    # Analisa a relevância diretamente (não através de similaridade vetorial)
    # Retorna um score no intervalo de 0 a 1 indicando a relevância
    def re_ranking(
            self,
            query: str,
            chunks_recuperados: List[str],
            indices_recuperados: List[int],
            cross_encoder: CrossEncoder,
            top_k_final: int = 16,
            threshold: float = 0.2
        ) -> Tuple[List[int], List[float]]:
        # Preparar pares [query, chunk] para o Cross-Encoder
        cross_encoder_input = [[query, chunk] for chunk in chunks_recuperados]

        # Calcular scores de relevância
        cross_scores = cross_encoder.predict(cross_encoder_input, show_progress_bar=True)
        
        # Combinar scores com índices e ordenar de maneira decrescente
        scored_results = list(zip(cross_scores, indices_recuperados))
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Aplicar threshold e pegar top_k
        filtered_results = [(score, idx) for score, idx in scored_results if score >= threshold]
        final_results = filtered_results[:top_k_final]

        if not final_results:
            final_results = scored_results[:min(3, len(scored_results))]

        final_scores = [score for score, idx in final_results]
        final_indices = [idx for score, idx in final_results]

        return final_indices, final_scores


    # Usamos os melhores trechos (chunks) como contexto para gerar a resposta final
    def answer_generation(
            self, 
            query: str, 
            chunk_indices: List[int], 
            chunk_scores: List[float],
            chunks: List[str],
            # model: str 
        ) -> str:

        # Construir o contexto com os chunks selecionados
        context_parts = []
        for i, (chunk_idx, score) in enumerate(zip(chunk_indices, chunk_scores)):
            context_parts.append(f"Fonte {i+1} (Chunk {chunk_idx}, Relevância: {score:.2f}):")
            context_parts.append(f'"""\n{chunks[chunk_idx]}\n"""')
            context_parts.append("")
        context = "\n".join(context_parts)

        # System Prompt para guiar a resposta:
        system_prompt = """
            Você é um assistente especializado em análise de documentos.

            Sua tarefa é responder à pergunta do usuário baseando-se EXCLUSIVAMENTE nos trechos
            de texto fornecidos como contexto.

            Instruções importantes:
            1. Use APENAS as informações presentes no contexto fornecido
            2. Se a informação estiver explícita, cite a fonte: [Fonte X]
            3. Se a informação não estiver explícita, você pode responder com partes relevantes dos trechos
            4. Se precisar sintetizar informações de múltiplas fontes, cite todas: [Fontes X, Y]
            5. Se não encontrar informação suficiente, diga que não foi possível responder com base no contexto
            6. Seja conciso e direto
            7. Não adicione conhecimento externo ao documento

            Responda de forma clara e bem fundamentada.
        """

        final_prompt = f"""
            CONTEXTO:
            {context}

            PERGUNTA: {query}

            RESPOSTA:
        """

        response = self.llm_api(final_prompt, system_prompt)
        return response if response else 'Não foi possível gerar uma resposta.'
    
    def full_rag_system(
            self, 
            chunks: List[str], 
            passage_embeddings: torch.Tensor, 
            embedding_model: SentenceTransformer,
            cross_encoder_model: CrossEncoder,
            question: str,
        ) -> dict:
        """
        Sistema RAG completo que processa uma pergunta e retorna uma resposta fundamentada.

        Args:
            pergunta: A pergunta do usuário

        Returns:
            Dicionário com a resposta e metadados do processo
        """

        result = {
            "pergunta": question,
            "resposta": "",
            "chunks_utilizados": [],
            "scores": [],
            "sucesso": False
        }

        if self.openrouter_api_key:
            multi_queries = self.multi_query_extention(question)
            hyde_doc = self.hyde_query_extention(question)
        else: 
            multi_queries = []
            hyde_doc = ''
        
        retrieved_chunks_local, retrieved_index_local, retrieval_scores_local = self.hybrid_search(
            chunks=chunks,
            query_original=question,
            multi_queries=multi_queries,
            hyde_doc=hyde_doc,
            embedding_model=embedding_model,
            passage_embeddings=passage_embeddings,
            top_k=self.top_k_retrieval
        )

        final_index_local, final_scores_local = self.re_ranking(
            query=question,
            chunks_recuperados=retrieved_chunks_local,
            cross_encoder=cross_encoder_model,
            indices_recuperados=retrieved_index_local,
            threshold=self.cross_encoder_threshold,
            top_k_final=self.top_k_rerank,
        )

        if not final_index_local:
            answer = 'Desculpe, não encontrei informações relevantes para responder à sua pergunta no documento.'
        else:
             if self.openrouter_api_key:
                answer = self.answer_generation(
                    query=question,
                    chunk_indices=final_index_local,
                    chunk_scores=final_scores_local,
                    chunks=chunks
                )

        result["resposta"] = answer
        result["chunks_utilizados"] = final_index_local
        result["scores"] = final_scores_local
        result["sucesso"] = True

        return result

    def simple_rag_system(
            self, 
            chunks: List[str], 
            passage_embeddings: torch.Tensor, 
            embedding_model: SentenceTransformer,
            question: str,
        ) -> dict:
            """
            Sistema RAG simples (sem Hybrid Search e sem ReRanking).
            Usa similaridade de cosseno para recuperar os chunks mais relevantes.
            """

            result = {
                "pergunta": question,
                "resposta": "",
                "chunks_utilizados": [],
                "scores": [],
                "sucesso": False
            }

            # Criar embedding da pergunta
            query_embedding = embedding_model.encode(
                f"query: {question}",
                convert_to_tensor=True,
                show_progress_bar=False,
                device=self.find_device()
            )

            # Similaridade de cosseno entre a pergunta e todos os chunks
            similarities = cosine_similarity(query_embedding, passage_embeddings)

            # Selecionar os top-k chunks mais relevantes
            top_scores, top_indices = torch.topk(similarities, k=min(self.top_k_retrieval, len(chunks)))

            top_indices = top_indices.cpu().tolist()
            top_scores = top_scores.cpu().tolist()

            # Geração de resposta
            if not top_indices:
                answer = "Desculpe, não encontrei informações relevantes para responder à sua pergunta no documento."
            else:
                if self.openrouter_api_key:
                    answer = self.answer_generation(
                        query=question,
                        chunk_indices=top_indices,
                        chunk_scores=top_scores,
                        chunks=chunks
                    )
                else:
                    answer = "Chave de API não configurada para geração de resposta."

            result["resposta"] = answer
            result["chunks_utilizados"] = top_indices
            result["scores"] = top_scores
            result["sucesso"] = True

            return result

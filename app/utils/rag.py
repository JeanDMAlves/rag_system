from pypdf import PdfReader
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple
import torch
import requests
import time
import os

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
            llm_generation_model = 'openai/gpt-4o'
        ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.cross_encoder_threshold = cross_encoder_threshold
        self.embedding_model = embedding_model
        self.cross_encoder_model = cross_encoder_model
        self.llm_generation_model = llm_generation_model
        self.openrouter_api_key = os.environ.get('OPENROUTER_API_KEY')

    def find_device(self) -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_document(self, path: str):
        if '.pdf' in path:
            raw_text = self.load_pdf(path)
            file_name = path.split('/')[-1]
            file_name = file_name.replace('.pdf', '.txt')
            # Salvar o texto em txt 
            self.save_text_from_string(raw_text, path_to_save=f'./txts/{file_name}')
            return raw_text
        elif '.txt' in path:
            return self.load_txt(path)
        raise Exception('Formato não suportado')

    # Ler o documento de texto (PDF)
    def load_pdf(self, file):
        reader = PdfReader(file)
        raw_text = ''
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text + '\n'
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
    
    def save_text_from_string(self, text: str, path_to_save: str):
        with open(path_to_save, 'w', encoding='utf-8') as f:
            f.write(text)
    
    # Dividir o documento em pedaços menores e gerenciáveis
    def chunking(self, text):
        tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
        
        # Transforma o texto em tokens
        tokens = tokenizer.encode(text)
        
        # Calcula a sobreposição dos tokens
        tokens_overlap = int(self.chunk_size * (self.chunk_overlap / 100))
        step = self.chunk_size - tokens_overlap

        text_chunks = []
        # Divide os tokens em chunks
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + self.chunk_size]
            
            if not chunk_tokens:
                continue
            
            text_chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            text_chunks.append(text_chunk)
        
        return text_chunks
    
    def chunking_rcts(self, text: str):
        """
        Divide o texto em pedaços coesos (sem quebrar palavras) usando RecursiveCharacterTextSplitter
        e garante que cada pedaço não ultrapasse o limite de tokens do modelo de embedding.
        """
        # 1. Split inicial com RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 4,   # usa caracteres como proxy (4 chars ≈ 1 token)
            chunk_overlap=int(self.chunk_size * self.chunk_overlap / 100) * 4,
            length_function=len,
        )
        initial_chunks = splitter.split_text(text)

        # 2. Ajustar cada pedaço com base no tokenizer
        final_chunks = []
        for chunk in initial_chunks:
            tokens = self.tokenizer.encode(chunk, add_special_tokens=False)

            # Se o chunk couber no limite, mantém
            if len(tokens) <= self.chunk_size:
                final_chunks.append(chunk)
                continue

            # Se for maior que o limite, divide em sub-chunks
            step = self.chunk_size - int(self.chunk_size * self.chunk_overlap / 100)
            for i in range(0, len(tokens), step):
                sub_tokens = tokens[i:i + self.chunk_size]
                sub_chunk = self.tokenizer.decode(sub_tokens, skip_special_tokens=True)
                if sub_chunk.strip():
                    final_chunks.append(sub_chunk)

        return final_chunks
    
    def load_embedding_model(self) -> SentenceTransformer:
        embedding_model = SentenceTransformer(self.embedding_model)
        device = self.find_device()
        if device == 'cuda':
            embedding_model = embedding_model.to(device)
        return embedding_model
    
    def load_cross_encoder(self) -> CrossEncoder:
        return CrossEncoder(self.cross_encoder_model)
    
    # Convertemos cada chunk em vetores numéricos que capturam o significado
    def embeddings(self, chunks, embedding_model: SentenceTransformer):
        passages_prefixed = [f"passage: {chunk}" for chunk in chunks]
        passage_embeddings = embedding_model.encode(
            passages_prefixed,
            convert_to_tensor = True,
            show_progress_bar = True,
            device = self.find_device(),
            batch_size = 32
        )
        return passage_embeddings

    def llm_api(self, prompt: str, system_prompt: str, max_retries: int = 3) -> str:
        # TODO: Subsituir self.llm_generation_model por uma lista de modelos, já que
        # Os modelos gratuitos tendem a dar error_code 429:
        # Resposta: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"deepseek/deepseek-chat-v3.1:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumula

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
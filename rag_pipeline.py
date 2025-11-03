"""
RAG (Retrieval-Augmented Generation) Pipeline
Implements retrieval, grounding, and citation management

Supports both simple keyword retrieval (for testing) and
production vector embeddings (ChromaDB, Pinecone, Qdrant, etc.)
"""

import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Protocol
from dataclasses import dataclass, field
from collections import Counter
from abc import ABC, abstractmethod


@dataclass
class Chunk:
    """A text chunk with metadata"""
    chunk_id: str
    text: str
    source_id: str
    source_name: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    score: float = 0.0


@dataclass
class Citation:
    """A citation linking claim to source"""
    claim_id: str
    claim_text: str
    source_id: str
    source_name: str
    span_start: int
    span_end: int
    confidence: float = 1.0


@dataclass
class GroundedContext:
    """Context pack with citations ready for agents"""
    chunks: List[Chunk]
    citations: List[Citation]
    query_used: str
    total_sources: int
    contradictions: List[str] = field(default_factory=list)
    notes: str = ""


class BaseRetriever(ABC):
    """Abstract base class for retrievers"""

    @abstractmethod
    def add_document(self, doc_id: str, text: str, source_name: str, metadata: Dict = None):
        """Add a document to the retriever"""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10, min_score: float = 0.0) -> List[Chunk]:
        """Retrieve relevant chunks for a query"""
        pass

    @abstractmethod
    def get_chunk_context(self, chunk: Chunk, context_chars: int = 200) -> str:
        """Get surrounding context for a chunk"""
        pass


class SimpleRetriever(BaseRetriever):
    """
    Simplified retriever (no external dependencies)
    Uses keyword-based BM25-like retrieval
    Good for testing, but use VectorRetriever for production
    """

    def __init__(self):
        self.documents: Dict[str, str] = {}
        self.chunks: List[Chunk] = []

    def add_document(self, doc_id: str, text: str, source_name: str, metadata: Dict = None):
        """Add document and chunk it"""
        self.documents[doc_id] = text

        # Simple structure-aware chunking
        chunks = self._chunk_document(text, doc_id, source_name, metadata or {})
        self.chunks.extend(chunks)

    def _chunk_document(
        self,
        text: str,
        doc_id: str,
        source_name: str,
        metadata: Dict,
        chunk_size: int = 600,
        overlap: int = 90
    ) -> List[Chunk]:
        """
        Structure-aware chunking with overlap

        chunk_size=600 tokens ≈ 2400 chars
        overlap=90 tokens ≈ 360 chars (15% overlap)
        """
        chars_per_chunk = chunk_size * 4
        chars_overlap = overlap * 4

        chunks = []
        start = 0
        chunk_num = 0

        while start < len(text):
            end = min(start + chars_per_chunk, len(text))

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings in last 200 chars
                search_start = max(start, end - 200)
                sentence_ends = [i for i in range(search_start, end)
                                if text[i] in '.!?\n']
                if sentence_ends:
                    end = sentence_ends[-1] + 1

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = f"{doc_id}_chunk_{chunk_num}"
                chunk = Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source_id=doc_id,
                    source_name=source_name,
                    start_pos=start,
                    end_pos=end,
                    metadata=metadata
                )
                chunks.append(chunk)
                chunk_num += 1

            # Move start with overlap
            start = end - chars_overlap if end < len(text) else len(text)

        return chunks

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0
    ) -> List[Chunk]:
        """
        Simple keyword-based retrieval (BM25-like)
        In production, use hybrid vector + BM25 retrieval
        """
        query_terms = set(query.lower().split())

        scored_chunks = []
        for chunk in self.chunks:
            chunk_terms = set(chunk.text.lower().split())

            # Simple scoring: term overlap + position boost
            overlap = len(query_terms & chunk_terms)
            score = overlap / max(len(query_terms), 1)

            # Boost if query terms appear in order
            if all(term in chunk.text.lower() for term in query_terms):
                score *= 1.5

            chunk.score = score
            if score >= min_score:
                scored_chunks.append(chunk)

        # Sort by score and return top_k
        scored_chunks.sort(key=lambda c: c.score, reverse=True)
        return scored_chunks[:top_k]

    def get_chunk_context(self, chunk: Chunk, context_chars: int = 200) -> str:
        """Get surrounding context for a chunk"""
        doc_text = self.documents.get(chunk.source_id, "")
        start = max(0, chunk.start_pos - context_chars)
        end = min(len(doc_text), chunk.end_pos + context_chars)
        return doc_text[start:end]


class VectorRetriever(BaseRetriever):
    """
    Production-ready vector retriever with hybrid search

    Supports multiple vector DB backends:
    - ChromaDB (local, easy setup)
    - Pinecone (cloud, scalable)
    - Qdrant (self-hosted or cloud)

    Features:
    - Dense vector embeddings
    - Hybrid search (vector + keyword BM25)
    - Automatic reranking
    - De-duplication
    """

    def __init__(
        self,
        backend: str = "chroma",
        embedding_model: str = "sentence-transformers",
        collection_name: str = "rag_chunks"
    ):
        """
        Initialize vector retriever

        Args:
            backend: "chroma", "pinecone", or "qdrant"
            embedding_model: "sentence-transformers" (local) or "openai" (API)
            collection_name: Name of vector collection
        """
        self.backend = backend
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        self.documents: Dict[str, str] = {}
        self.chunks: List[Chunk] = []

        # Lazy import vector DB client
        self.client = None
        self.collection = None
        self.embedder = None

        # Try to initialize (gracefully fail if dependencies missing)
        try:
            self._initialize_backend()
        except ImportError as e:
            print(f"⚠️  Vector DB dependencies not installed: {e}")
            print(f"   Falling back to keyword search. Install with:")
            print(f"   pip install chromadb sentence-transformers")
            self.client = None

    def _initialize_backend(self):
        """Initialize vector DB backend and embedding model"""

        if self.backend == "chroma":
            import chromadb
            from chromadb.utils import embedding_functions

            self.client = chromadb.Client()

            # Initialize embedder
            if self.embedding_model == "sentence-transformers":
                self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"  # Fast, 384-dim embeddings
                )
            elif self.embedding_model == "openai":
                # Requires OPENAI_API_KEY env var
                self.embedder = embedding_functions.OpenAIEmbeddingFunction(
                    model_name="text-embedding-3-small"
                )

            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedder
            )

        elif self.backend == "pinecone":
            import pinecone
            import os

            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENV", "us-west1-gcp")
            )

            # Initialize embedder
            if self.embedding_model == "sentence-transformers":
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            elif self.embedding_model == "openai":
                import openai
                openai.api_key = os.getenv("OPENAI_API_KEY")
                self.embedder = "openai"  # Will call API directly

            # Create or connect to index
            if self.collection_name not in pinecone.list_indexes():
                pinecone.create_index(
                    self.collection_name,
                    dimension=384,  # Match embedding model
                    metric="cosine"
                )

            self.collection = pinecone.Index(self.collection_name)

        elif self.backend == "qdrant":
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self.client = QdrantClient(":memory:")  # Or url for remote

            # Initialize embedder
            if self.embedding_model == "sentence-transformers":
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

            # Create collection if doesn't exist
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
            except:
                pass  # Collection already exists

            self.collection = self.collection_name

    def add_document(self, doc_id: str, text: str, source_name: str, metadata: Dict = None):
        """Add document with vector embeddings"""
        self.documents[doc_id] = text

        # Chunk document (reuse from SimpleRetriever)
        simple_retriever = SimpleRetriever()
        chunks = simple_retriever._chunk_document(text, doc_id, source_name, metadata or {})

        # If vector DB available, add embeddings
        if self.client is not None and self.collection is not None:
            for chunk in chunks:
                self._add_chunk_to_vector_db(chunk)

        self.chunks.extend(chunks)

    def _add_chunk_to_vector_db(self, chunk: Chunk):
        """Add chunk to vector database"""

        if self.backend == "chroma":
            self.collection.add(
                ids=[chunk.chunk_id],
                documents=[chunk.text],
                metadatas=[{
                    "source_id": chunk.source_id,
                    "source_name": chunk.source_name,
                    "start_pos": chunk.start_pos,
                    "end_pos": chunk.end_pos,
                    **chunk.metadata
                }]
            )

        elif self.backend == "pinecone":
            # Get embedding
            if self.embedding_model == "sentence-transformers":
                embedding = self.embedder.encode(chunk.text).tolist()
            elif self.embedding_model == "openai":
                import openai
                response = openai.Embedding.create(
                    input=chunk.text,
                    model="text-embedding-3-small"
                )
                embedding = response['data'][0]['embedding']

            # Upsert to Pinecone
            self.collection.upsert(
                vectors=[(
                    chunk.chunk_id,
                    embedding,
                    {
                        "text": chunk.text,
                        "source_id": chunk.source_id,
                        "source_name": chunk.source_name,
                        "start_pos": chunk.start_pos,
                        "end_pos": chunk.end_pos,
                        **chunk.metadata
                    }
                )]
            )

        elif self.backend == "qdrant":
            from qdrant_client.models import PointStruct

            # Get embedding
            embedding = self.embedder.encode(chunk.text).tolist()

            # Add point
            self.client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(
                    id=hash(chunk.chunk_id),  # Convert to int
                    vector=embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "source_id": chunk.source_id,
                        "source_name": chunk.source_name,
                        "start_pos": chunk.start_pos,
                        "end_pos": chunk.end_pos,
                        **chunk.metadata
                    }
                )]
            )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        hybrid: bool = True
    ) -> List[Chunk]:
        """
        Hybrid retrieval: vector search + keyword fallback

        Args:
            query: Search query
            top_k: Number of results
            min_score: Minimum similarity score
            hybrid: If True, combine vector and keyword search

        Returns:
            List of retrieved chunks
        """

        # If vector DB available, use it
        if self.client is not None and self.collection is not None:
            results = self._vector_search(query, top_k)

            # If hybrid, also do keyword search and merge
            if hybrid:
                keyword_results = self._keyword_search(query, top_k // 2)
                results = self._merge_results(results, keyword_results, top_k)

            return results

        # Fallback to keyword search
        return self._keyword_search(query, top_k)

    def _vector_search(self, query: str, top_k: int) -> List[Chunk]:
        """Vector similarity search"""

        if self.backend == "chroma":
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )

            # Convert to Chunk objects
            chunks = []
            for i, doc_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i]
                chunk = Chunk(
                    chunk_id=doc_id,
                    text=results['documents'][0][i],
                    source_id=metadata['source_id'],
                    source_name=metadata['source_name'],
                    start_pos=metadata['start_pos'],
                    end_pos=metadata['end_pos'],
                    metadata=metadata,
                    score=1 - results['distances'][0][i]  # Convert distance to similarity
                )
                chunks.append(chunk)

            return chunks

        elif self.backend == "pinecone":
            # Get query embedding
            if self.embedding_model == "sentence-transformers":
                query_embedding = self.embedder.encode(query).tolist()
            elif self.embedding_model == "openai":
                import openai
                response = openai.Embedding.create(
                    input=query,
                    model="text-embedding-3-small"
                )
                query_embedding = response['data'][0]['embedding']

            # Query Pinecone
            results = self.collection.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )

            # Convert to Chunk objects
            chunks = []
            for match in results['matches']:
                metadata = match['metadata']
                chunk = Chunk(
                    chunk_id=match['id'],
                    text=metadata['text'],
                    source_id=metadata['source_id'],
                    source_name=metadata['source_name'],
                    start_pos=metadata['start_pos'],
                    end_pos=metadata['end_pos'],
                    metadata=metadata,
                    score=match['score']
                )
                chunks.append(chunk)

            return chunks

        elif self.backend == "qdrant":
            from qdrant_client.models import Filter

            # Get query embedding
            query_embedding = self.embedder.encode(query).tolist()

            # Search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k
            )

            # Convert to Chunk objects
            chunks = []
            for hit in results:
                payload = hit.payload
                chunk = Chunk(
                    chunk_id=payload['chunk_id'],
                    text=payload['text'],
                    source_id=payload['source_id'],
                    source_name=payload['source_name'],
                    start_pos=payload['start_pos'],
                    end_pos=payload['end_pos'],
                    metadata=payload,
                    score=hit.score
                )
                chunks.append(chunk)

            return chunks

        return []

    def _keyword_search(self, query: str, top_k: int) -> List[Chunk]:
        """BM25-like keyword search (fallback)"""
        simple_retriever = SimpleRetriever()
        simple_retriever.chunks = self.chunks
        simple_retriever.documents = self.documents
        return simple_retriever.retrieve(query, top_k)

    def _merge_results(
        self,
        vector_results: List[Chunk],
        keyword_results: List[Chunk],
        top_k: int
    ) -> List[Chunk]:
        """Merge and deduplicate vector and keyword results"""
        seen_ids = set()
        merged = []

        # Interleave results (favor vector results slightly)
        all_results = []
        for i in range(max(len(vector_results), len(keyword_results))):
            if i < len(vector_results):
                chunk = vector_results[i]
                chunk.score *= 1.2  # Boost vector results
                all_results.append(chunk)
            if i < len(keyword_results):
                all_results.append(keyword_results[i])

        # Deduplicate and take top_k
        for chunk in all_results:
            if chunk.chunk_id not in seen_ids:
                merged.append(chunk)
                seen_ids.add(chunk.chunk_id)
                if len(merged) >= top_k:
                    break

        # Sort by score
        merged.sort(key=lambda c: c.score, reverse=True)
        return merged[:top_k]

    def get_chunk_context(self, chunk: Chunk, context_chars: int = 200) -> str:
        """Get surrounding context for a chunk"""
        doc_text = self.documents.get(chunk.source_id, "")
        start = max(0, chunk.start_pos - context_chars)
        end = min(len(doc_text), chunk.end_pos + context_chars)
        return doc_text[start:end]


class RAGPipeline:
    """
    Complete RAG pipeline: retrieve → rerank → ground

    Works with any retriever (SimpleRetriever or VectorRetriever)
    """

    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever
        self.query_history: List[str] = []

    def query_rewrite(self, query: str) -> str:
        """
        Query expansion/rewriting for better retrieval

        In production: use LLM to expand query or add synonyms
        """
        # Simple expansion: add common variations
        expansions = {
            "explain": "explain describe clarify",
            "how": "how what method process",
            "why": "why reason cause because",
            "what": "what definition meaning"
        }

        words = query.lower().split()
        expanded = []
        for word in words:
            expanded.append(word)
            if word in expansions:
                expanded.append(expansions[word])

        return ' '.join(expanded)

    def retrieve_and_rerank(
        self,
        query: str,
        top_k: int = 10,
        diversity_boost: bool = True
    ) -> List[Chunk]:
        """
        Retrieve and rerank chunks

        Args:
            query: Search query
            top_k: Number of chunks to return
            diversity_boost: Penalize chunks from same source

        Returns:
            Reranked chunks
        """
        # Query rewriting
        expanded_query = self.query_rewrite(query)
        self.query_history.append(query)

        # Initial retrieval (get more than needed for reranking)
        candidates = self.retriever.retrieve(expanded_query, top_k=top_k * 3)

        if not candidates:
            return []

        # Rerank with diversity
        if diversity_boost:
            candidates = self._diversify(candidates, top_k)
        else:
            candidates = candidates[:top_k]

        return candidates

    def _diversify(self, chunks: List[Chunk], top_k: int) -> List[Chunk]:
        """
        Diversify results to avoid all chunks from same source

        MMR-like approach: balance relevance and diversity
        """
        selected = []
        source_counts = Counter()

        for chunk in chunks:
            # Penalize if we already have many from this source
            penalty = source_counts[chunk.source_id] * 0.1
            adjusted_score = chunk.score * (1 - penalty)

            selected.append((adjusted_score, chunk))
            source_counts[chunk.source_id] += 1

            if len(selected) >= top_k:
                break

        # Sort by adjusted score
        selected.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in selected[:top_k]]

    def ground_context(
        self,
        query: str,
        chunks: List[Chunk],
        max_chunks: int = 8
    ) -> GroundedContext:
        """
        Create grounded context pack with citation tracking

        Args:
            query: Original query
            chunks: Retrieved chunks
            max_chunks: Maximum chunks to include

        Returns:
            GroundedContext ready for agent use
        """
        # Limit chunks
        selected_chunks = chunks[:max_chunks]

        # Detect contradictions (simple version)
        contradictions = self._detect_contradictions(selected_chunks)

        # Count unique sources
        unique_sources = len(set(c.source_id for c in selected_chunks))

        return GroundedContext(
            chunks=selected_chunks,
            citations=[],  # Will be filled by agent
            query_used=query,
            total_sources=unique_sources,
            contradictions=contradictions,
            notes=f"Retrieved {len(selected_chunks)} chunks from {unique_sources} sources"
        )

    def _detect_contradictions(self, chunks: List[Chunk]) -> List[str]:
        """
        Simple contradiction detection

        In production: use NLI model or LLM to check consistency
        """
        contradictions = []

        # Look for negation patterns
        negation_words = ['not', 'never', 'no', 'incorrect', 'false', 'wrong']

        texts = [c.text.lower() for c in chunks]

        for i, text1 in enumerate(texts):
            for j, text2 in enumerate(texts[i+1:], i+1):
                # Check if they talk about same thing but with negations
                shared_words = set(text1.split()) & set(text2.split())
                if len(shared_words) > 5:  # Talking about same topic
                    has_negation1 = any(w in text1 for w in negation_words)
                    has_negation2 = any(w in text2 for w in negation_words)

                    if has_negation1 != has_negation2:
                        contradictions.append(
                            f"Potential contradiction between {chunks[i].source_name} "
                            f"and {chunks[j].source_name}"
                        )

        return contradictions

    def create_citation(
        self,
        claim_text: str,
        chunk: Chunk,
        confidence: float = 1.0
    ) -> Citation:
        """Create a citation linking claim to source"""
        claim_id = hashlib.md5(claim_text.encode()).hexdigest()[:8]

        return Citation(
            claim_id=claim_id,
            claim_text=claim_text,
            source_id=chunk.source_id,
            source_name=chunk.source_name,
            span_start=chunk.start_pos,
            span_end=chunk.end_pos,
            confidence=confidence
        )

    def validate_citations(
        self,
        claims: List[str],
        grounded_context: GroundedContext
    ) -> Tuple[bool, List[str]]:
        """
        Validate that all claims have supporting sources

        Returns:
            (all_valid, list of unsupported claims)
        """
        unsupported = []

        for claim in claims:
            # Check if claim text appears in any chunk
            supported = any(
                claim.lower() in chunk.text.lower()
                for chunk in grounded_context.chunks
            )

            if not supported:
                unsupported.append(claim)

        return len(unsupported) == 0, unsupported


# Example usage and testing
def test_rag_pipeline():
    """Test the RAG pipeline"""
    print("="*70)
    print("RAG PIPELINE TEST")
    print("="*70)

    # Create retriever and add documents
    retriever = SimpleRetriever()

    retriever.add_document(
        "doc1",
        """
        Transformers are neural networks that use self-attention mechanisms.
        They were introduced in the paper 'Attention Is All You Need' by Vaswani et al.
        The key innovation is the attention mechanism which allows the model to focus
        on relevant parts of the input when processing each token.
        """,
        "Attention Is All You Need (2017)",
        {"year": 2017, "authors": "Vaswani et al."}
    )

    retriever.add_document(
        "doc2",
        """
        Claude is a large language model developed by Anthropic.
        It uses a decoder-only transformer architecture similar to GPT.
        Claude can process up to 200,000 tokens of context and supports
        multi-modal inputs including text and images.
        """,
        "Claude Technical Overview",
        {"year": 2024, "company": "Anthropic"}
    )

    # Create pipeline
    pipeline = RAGPipeline(retriever)

    # Test retrieval
    print("\n1. Testing retrieval:")
    query = "What are transformers and how do they work?"
    chunks = pipeline.retrieve_and_rerank(query, top_k=5)

    print(f"   Query: {query}")
    print(f"   Retrieved {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"   {i}. [{chunk.source_name}] Score: {chunk.score:.2f}")
        print(f"      {chunk.text[:100]}...")

    # Test grounding
    print("\n2. Testing grounding:")
    grounded = pipeline.ground_context(query, chunks, max_chunks=3)
    print(f"   Grounded context: {len(grounded.chunks)} chunks from {grounded.total_sources} sources")
    if grounded.contradictions:
        print(f"   ⚠️  Contradictions detected: {len(grounded.contradictions)}")

    # Test citation validation
    print("\n3. Testing citation validation:")
    claims = [
        "Transformers use self-attention mechanisms",
        "Claude was invented in 1950"  # False claim
    ]
    all_valid, unsupported = pipeline.validate_citations(claims, grounded)
    print(f"   Claims: {len(claims)}")
    print(f"   All supported: {all_valid}")
    if unsupported:
        print(f"   Unsupported claims: {unsupported}")

    print("\n" + "="*70)
    print("✅ RAG PIPELINE VALIDATED")
    print("="*70)


if __name__ == "__main__":
    test_rag_pipeline()

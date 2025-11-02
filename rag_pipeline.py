"""
RAG (Retrieval-Augmented Generation) Pipeline
Implements retrieval, grounding, and citation management
"""

import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter


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


class SimpleRetriever:
    """
    Simplified retriever (no external dependencies)
    In production, replace with vector DB (Pinecone, Weaviate, etc.)
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


class RAGPipeline:
    """Complete RAG pipeline: retrieve → rerank → ground"""

    def __init__(self, retriever: SimpleRetriever):
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

"""Builds grounded context strings for LLM prompts."""

from recruitment_ai.reranking.reranker import RerankResult


class ContextBuilder:
    """Assembles retrieved document chunks into clean, cited prompts."""

    def build_context(self, retrieved_chunks: list[RerankResult]) -> str:
        """Format chunks into numbered context blocks with provenance metadata."""
        if not retrieved_chunks:
            return "No relevant context found."

        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 1)
            doc_type = chunk.metadata.get("document_type", "document")
            header = (
                f"[Document {i} | Type: {doc_type} | Source: {source} (Page {page})]"
            )
            context_blocks.append(f"{header}\n{chunk.text}")

        return "\n\n---\n\n".join(context_blocks)

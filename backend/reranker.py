from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Union


class Reranker:
    def __init__(self):
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(self, query: str, docs: List[Union[str, Dict[str, Any]]], top_k: int = 5) -> List[Union[str, Dict[str, Any]]]:
        """
        Rerank documents using cross-encoder
        
        Args:
            query: Search query
            docs: List of documents (can be strings OR dicts with 'text' field)
            top_k: Number of top results to return
            
        Returns:
            Reranked list of documents in same format as input
        """
        if not docs:
            return []
        
        # Extract text for scoring (handle both strings and dicts)
        texts = []
        for doc in docs:
            if isinstance(doc, dict):
                texts.append(doc.get("text", ""))
            else:
                texts.append(str(doc))
        
        # Create query-document pairs
        pairs = [(query, text) for text in texts]
        
        # Get reranking scores
        scores = self.model.predict(pairs)
        
        # Add rerank scores to dict documents
        for i, doc in enumerate(docs):
            if isinstance(doc, dict):
                doc["rerank_score"] = float(scores[i])
        
        # Sort by rerank score
        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top k documents (preserve original format)
        return [doc for doc, _ in ranked[:top_k]]
"""Internal RAG system - Agent memory/cognition (NOT a tool)."""

from .retriever import MedicalKnowledgeRetriever, get_retriever

__all__ = ["MedicalKnowledgeRetriever", "get_retriever"]


from src.rag_pipeline import RAGPipeline


class RAGService:
    def __init__(self):
        self.pipeline = RAGPipeline()

    def ask(self, question: str, top_k: int = 5):
        result = self.pipeline.answer_question(question=question, top_k=top_k)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "confidence": result["confidence"],
            "sources": result["sources"],
        }

    def retrieve(self, question: str, top_k: int = 5):
        sources = self.pipeline.retriever.retrieve(question=question, top_k=top_k)

        return {
            "question": question,
            "sources": sources,
        }

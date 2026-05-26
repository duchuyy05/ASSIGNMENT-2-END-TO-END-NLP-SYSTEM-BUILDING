import os
import sys

from src.rag_pipeline import RAGPipeline


QUESTIONS_PATH = "data/test/questions.txt"
OUTPUT_PATH = "system_outputs/system_output_1.txt"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    os.makedirs("system_outputs", exist_ok=True)

    rag = RAGPipeline()

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for question in questions:
            result = rag.answer_question(question)
            answer = result["answer"]

            f.write(answer + "\n")

            print(f"Q: {question}")
            print(f"A: {answer}")
            print("-" * 50)

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

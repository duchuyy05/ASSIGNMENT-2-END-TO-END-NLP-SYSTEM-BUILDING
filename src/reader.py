import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from src.config import READER_MODEL


class DistilBertReader:
    def __init__(self, model_name: str = READER_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()

    def answer(self, question: str, context: str) -> dict:
        if not context.strip():
            return {
                "answer": "",
                "score": 0.0,
            }

        encoded = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=512,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0]
        sequence_ids = encoded.sequence_ids(0)

        with torch.no_grad():
            outputs = self.model(**encoded)

        start_probs = torch.softmax(outputs.start_logits, dim=-1)
        end_probs = torch.softmax(outputs.end_logits, dim=-1)

        context_token_indices = [
            idx
            for idx, sequence_id in enumerate(sequence_ids)
            if sequence_id == 1 and tuple(offsets[idx].tolist()) != (0, 0)
        ]

        best_score = 0.0
        best_span = None
        max_answer_tokens = 30
        for start_idx in context_token_indices:
            for end_idx in context_token_indices:
                if end_idx < start_idx:
                    continue
                if end_idx - start_idx + 1 > max_answer_tokens:
                    break
                score = float(start_probs[0, start_idx] * end_probs[0, end_idx])
                if score > best_score:
                    best_score = score
                    best_span = (start_idx, end_idx)

        if best_span is None:
            return {"answer": "", "score": 0.0}

        start_idx, end_idx = best_span
        input_ids = encoded["input_ids"][0]
        answer_ids = input_ids[start_idx : end_idx + 1]
        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

        return {"answer": answer, "score": best_score}

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "google/flan-t5-base"

# Load tokenizer and seq2seq model directly
# (transformers v5 removed the text2text-generation pipeline)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()


def generate_answer(query: str, context: str) -> str:
    prompt = (
        "You are an AI research assistant. "
        "Answer the question based ONLY on the context below. "
        "If the answer is not found in the context, say 'Not enough information.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True,
        )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer.strip()
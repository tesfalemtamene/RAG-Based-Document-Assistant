
from pdf_loader import load_pdf
from text_chunker import chunk_text, clean_chunks
from vector_store import create_vector_store, similarity_search
from llm_model import generate_answer


def main():

    print("Loading PDF...")
    text = load_pdf("../data/sample_paper.pdf")

    print("Extracting text completed.")
    print("Total characters:", len(text))

    print("\nChunking text...")
    chunks = chunk_text(text)
    chunks = clean_chunks(chunks)

    print("Total chunks created:", len(chunks))

    # Create vector DB
    index, embeddings = create_vector_store(chunks)

    # User query
    query = input("\nEnter your question: ")

    # Retrieve relevant chunks
    results = similarity_search(query, index, chunks)

    # Combine context
    context = "\n\n".join(results)

    print("\nGenerating answer...\n")

    # Generate answer using LLM
    answer = generate_answer(query, context)

    print("Final Answer:\n")
    print(answer)


if __name__ == "__main__":
    main()
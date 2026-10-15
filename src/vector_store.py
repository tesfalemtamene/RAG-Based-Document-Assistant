from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
# MiniLM	is fast but MPNet is more accurate
# model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('all-mpnet-base-v2')

def create_vector_store(chunks):
    print("Generating embeddings...")

    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]

    print("Creating FAISS index...")

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Save after adding vectors so the file is not empty
    faiss.write_index(index, "vector_index.faiss")

    return index, embeddings


def similarity_search(query, index, chunks, top_k=3):
    print("Searching similar documents...")

    query_embedding = model.encode([query])

    distances, indices = index.search(np.array(query_embedding), top_k)

    results = []

    for i, idx in enumerate(indices[0]):
        results.append((chunks[idx], distances[0][i]))

    return results
    

    


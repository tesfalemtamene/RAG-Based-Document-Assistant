from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_text(text)

    return chunks
def clean_chunks(chunks):

    filtered_chunks = []

    for chunk in chunks:
        if len(chunk) > 50:   # remove near-empty chunks only
            filtered_chunks.append(chunk)

    return filtered_chunks
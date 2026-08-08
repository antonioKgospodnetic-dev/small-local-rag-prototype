import sys
import argparse
from pathlib import Path
import re
import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)
  
DATABASE_PATH = Path(__file__).resolve().parent / "chroma_database"
COLLECTION_NAME = "wuthering_heights"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
  
def ParseArgs(args):
    # cmd_params = {"--query", "--num_context_chunks"}
    parser = argparse.ArgumentParser(
        description="Expand a query before sending it to the RAG system."
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The query that should be expanded."
    )
    
    parser.add_argument(
        "--num_context_chunks",
        type=int,
        required=True,
        help="The amount the chunks that should be added as context."
    )

    return parser.parse_args(args)
  
def GetPromptContext(query, num_context_chunks):
    # This must match the model used to embed the book chunks.
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    
    client = chromadb.PersistentClient(
        path=str(DATABASE_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )
    
    # Stage 1: Semantic retrieval.
    results = collection.query(
        query_texts=[query],
        n_results=num_context_chunks,
        include=["documents", "distances", "metadatas"]
    )

    retrieved_ids = results["ids"][0]

    # Stage 2: Expand every retrieved chunk with its immediate neighbors.
    context_ids = set()

    for chunk_id, metadata in zip(
        retrieved_ids,
        results["metadatas"][0]
    ):
        context_ids.add(chunk_id)

        previous_id = metadata["previous_chunk_id"]
        if previous_id != "":
            context_ids.add(previous_id)

        next_id = metadata["next_chunk_id"]
        if next_id != "":
            context_ids.add(next_id)

    # Retrieve the semantic hits + their neighbors.
    expanded_results = collection.get(
        ids=list(context_ids),
        include=["documents", "metadatas"]
    )
    
    # Put chunks back into their original document order.
    expanded_chunks = []

    for document, metadata in zip(
        expanded_results["documents"],
        expanded_results["metadatas"]
    ):
        expanded_chunks.append({
            "document": document,
            "chunk_index": metadata["chunk_index"]
        })

    expanded_chunks.sort(
        key=lambda chunk: chunk["chunk_index"]
    )

    retrieved_context = "\n\n".join(
        chunk["document"]
        for chunk in expanded_chunks
    )

    return retrieved_context


#py Context_provider.py --query "How long did Cathy stay at the Thrushcross Grange?" --num_context_chunks 2
def main():
    args = ParseArgs(sys.argv[1:])
    
    retrieved_context = GetPromptContext(args.query, args.num_context_chunks)
    
    print(f"Retrieved context:\n{retrieved_context}\nQuestion:\n{args.query}")
    
if __name__ == "__main__":
    main()
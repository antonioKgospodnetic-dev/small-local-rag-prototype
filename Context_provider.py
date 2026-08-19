import sys
import argparse
from pathlib import Path
import re
import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction
)
  

# Validation of inputs
SUPPORTED_EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2"
]
  
def PositiveContextChunkCount(value):
    value = int(value)

    if value < 1:
        raise argparse.ArgumentTypeError(
            "Number of context chunks must be atleast 1."
        )

    return value

def ValidNeighborChunkCount(value):
    value = int(value)

    if value < 0:
        raise argparse.ArgumentTypeError(
            "Number of neighboring chunks must not be negative."
        )

    return value
  
def ValidDatabaseName(value):
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,510}[a-z0-9]", value):
        raise argparse.ArgumentTypeError(
            "Database name must be 3-512 characters, "
            "start and end with a lowercase letter or number, "
            "and contain only lowercase letters, numbers, '.', '_' and '-'."
        )

    if ".." in value:
        raise argparse.ArgumentTypeError(
            "Database name may not contain consecutive dots."
        )

    return value
  
def ValidateEmbeddingModel(model):
    if model not in SUPPORTED_EMBEDDING_MODELS:
        raise ValueError(
            f"Unsupported embedding model: '{model}'."
        )
  
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
        "--database_name",
        type=ValidDatabaseName,
        required=True,
        help="Name of the ChromaDB collection to query."
    )
    
    parser.add_argument(
        "--embedding_model",
        type=str,
        choices=SUPPORTED_EMBEDDING_MODELS,
        default="all-MiniLM-L6-v2",
        help="Sentence Transformer model used to create embeddings."
    )

    parser.add_argument(
        "--num_context_chunks",
        type=PositiveContextChunkCount,
        default=2,
        required=False,
        help="The amount the chunks that should be added as context."
    )
    
    parser.add_argument(
        "--range_neighbor_chunks",
        type=ValidNeighborChunkCount,
        default=1,
        required=False,
        help="Number of neighboring chunks to add on each side of every retrieved chunk."
    )

    return parser.parse_args(args)
  
  
def Add_neighboring_chunks(collection, context_ids, previous_id, next_id, remaining_range):
    if remaining_range <= 0:
        return

    new_previous_id = ""
    new_next_id = ""

    if previous_id != "":
        result = collection.get(
            ids=[previous_id],
            include=["metadatas"]
        )

        context_ids.add(previous_id)

        new_previous_id = (
            result["metadatas"][0]["previous_chunk_id"]
        )

    if next_id != "":
        result = collection.get(
            ids=[next_id],
            include=["metadatas"]
        )

        context_ids.add(next_id)

        new_next_id = (
            result["metadatas"][0]["next_chunk_id"]
        )

    Add_neighboring_chunks(collection, context_ids, new_previous_id, new_next_id, remaining_range - 1)
  
def GetPromptContext(collection_name, embedding_model, query, num_context_chunks, range_neighbor_chunks):  
    #Checks
    ValidateEmbeddingModel(embedding_model)
    
    if num_context_chunks < 1:
        raise ValueError(
            "num_context_chunks must be atleast 1."
        )
        
    if range_neighbor_chunks < 0:
        raise ValueError(
            "range_neighbor_chunks must not be a negative number."
        )
      
    # This must match the model used to embed the book chunks.
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=embedding_model
    )
    
    database_path = (
        Path(__file__).resolve().parent
        / f"chroma_database_{collection_name}"
    )
    
    client = chromadb.PersistentClient(
        path=str(database_path)
    )

    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
    except ValueError as e:
        raise ValueError(
            f"ChromaDB collection '{collection_name}' does not exist."
        ) from e
    
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

        Add_neighboring_chunks(
            collection,
            context_ids,
            metadata["previous_chunk_id"],
            metadata["next_chunk_id"],
            range_neighbor_chunks
        )


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


#py Context_provider.py --query "How long did Cathy stay at the Thrushcross Grange?" --database_name wuthering_heights --embedding_model "all-mpnet-base-v2"
#py Context_provider.py --query "How long did Cathy stay at the Thrushcross Grange?" --database_name wuthering_heights --num_context_chunks 2 
#py Context_provider.py --query "How long did Cathy stay at the Thrushcross Grange?" --database_name wuthering_heights --num_context_chunks 1 --range_neighbor_chunks 3
def main():
    args = ParseArgs(sys.argv[1:])
    
    database_name = args.database_name
    embedding_model = args.embedding_model
    query = args.query
    fixed_chunk_size = args.num_context_chunks
    range_neighbor_chunks = args.range_neighbor_chunks
    
    try:
        retrieved_context = GetPromptContext(database_name, embedding_model, query, fixed_chunk_size, range_neighbor_chunks)

    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    print(f"Retrieved context:\n{retrieved_context}\nQuestion:\n{args.query}")
    
if __name__ == "__main__":
    main()
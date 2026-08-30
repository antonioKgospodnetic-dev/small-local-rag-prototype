import sys
import argparse
import re
from openai import OpenAI
from Context_provider import GetPromptContext, FormatPassages
from RAG_constructor import SUPPORTED_EMBEDDING_MODELS

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

def ParseArgs(args):
    # cmd_params = {"--query", "--num_context_chunks"}
    parser = argparse.ArgumentParser(
        description="Expand a query before sending it to the RAG system."
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
        "--query",
        type=str,
        required=True,
        help="The query that should be expanded."
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

def GetRAGsAnswer(collection_name, embedding_model, query, num_context_chunks, range_neighbor_chunks):
    
    retrieved_context_passages = GetPromptContext(collection_name, embedding_model, query, num_context_chunks, range_neighbor_chunks)
    retrieved_context = FormatPassages(retrieved_context_passages)
    
    llm_client = OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio"
    )

    model_id = "gemma-3-270m-it-qat"
    print("Using model:", model_id)

    completion = llm_client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": (
                    "Form a single sentence answer to the question using only the provided context.\n"
                    "If the context doesnt explicitly contain the answer, respond exactly: "
                    "'The answer is not stated in the retrieved context.' "
                )
            },
            {
                "role": "user",
                "content": f"CONTEXT:\n{retrieved_context}\nQUESTION:\n{query}"
            }
        ],
        temperature=0.1,
        max_tokens=100
    )

    answer = completion.choices[0].message.content
    
    return retrieved_context, answer


#py Prompt_sender.py --query "How long did Cathy stay at the Thrushcross Grange?" --database_name wuthering_heights --embedding_model "all-mpnet-base-v2"
#py Prompt_sender.py --query "How long did Cathy stay at the Thrushcross Grange?" --database_name wuthering_heights --num_context_chunks 2 --range_neighbor_chunks 2
#py Prompt_sender.py --query "What did Alice find on a little glass table?" --database_name alice --num_context_chunks 2 --range_neighbor_chunks 3
#py Prompt_sender.py --query "Why does Alice not like the look of her sister’s book?" --database_name alice --num_context_chunks 2
#py Prompt_sender.py --query "What is the handbook about?" --database_name handbook --num_context_chunks 2
#py Prompt_sender.py --query "What is the definition of the concept called 'Overcover age - rate'?" --database_name handbook --num_context_chunks 2  
#py Prompt_sender.py --query "What was the unemployment rate in cities compared with rural areas in 2023?" --database_name statistics --num_context_chunks 1
def main():
    args = ParseArgs(sys.argv[1:])
    
    query = args.query
    num_context_chunks = args.num_context_chunks
    collection_name = args.database_name
    embedding_model = args.embedding_model
    range_neighbor_chunks = args.range_neighbor_chunks

    try:
        retrieved_context, answer = GetRAGsAnswer(collection_name, embedding_model, query, num_context_chunks, range_neighbor_chunks)

    except Exception as e:
        print(f"RAG failed: {e}")
        return 1

    print("\nQuestion:")
    print(query)

    print("\nRetrieved context:")
    print(retrieved_context)

    print("\nGenerated answer:")
    print(answer)
        
# Specifically built around expanding queries for vector base for Wurthering Heights
if __name__ == "__main__":
    main()
import sys
import argparse
import re
from openai import OpenAI
from Context_provider import GetPromptContext


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

    parser.add_argument(
        "--database_name",
        type=ValidDatabaseName,
        required=True,
        help="Name of the ChromaDB collection to query."
    )

    return parser.parse_args(args)

#py Prompt_sender.py --query "How long did Cathy stay at the Thrushcross Grange?" --num_context_chunks 2
def main():
    args = ParseArgs(sys.argv[1:])
    
    query = args.query
    num_context_chunks = args.num_context_chunks
    collection_name = args.database_name
    
    if ((num_context_chunks < 1) or (num_context_chunks > 2500)):
        print("err: Incorrect fixed chunk size")
        exit(1)
    
    try:
        retrieved_context = GetPromptContext(query, num_context_chunks, collection_name)
    except:
        print("Err: Running query expansion function failed")
        exit(1)
    
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
                    "Form a single sentence answer to the question using only the provided context"
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

    print("\nQuestion:")
    print(query)

    print("\nRetrieved context:")
    print(retrieved_context)

    print("\nGenerated answer:")
    print(answer)
        
# Specifically built around expanding queries for vector base for Wurthering Heights
if __name__ == "__main__":
    main()
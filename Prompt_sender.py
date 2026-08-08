import sys
import argparse
from openai import OpenAI
from Context_provider import GetPromptContext

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

#py Prompt_sender.py --query "How long did Cathy stay at the Thrushcross Grange?" --num_context_chunks 2
def main():
    args = ParseArgs(sys.argv[1:])
    query = args.query
    num_context_chunks = args.num_context_chunks
    
    try:
        retrieved_context = GetPromptContext(query, num_context_chunks)
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
import sys
import argparse
import csv
from pathlib import Path
from RAG_constructor import ConstructRAG, SUPPORTED_EMBEDDING_MODELS
from Prompt_sender import GetRAGsAnswer
import time


DATABASE_PATHS = {
    "alice": "example_PDFs/Alice_in_wonderland.pdf",
    "wuthering_heights": "example_PDFs/Wuthering heights.pdf",
    "handbook": "example_PDFs/Handbook for quality and metadata reports ESS.pdf",
    "statistics": "example_PDFs/Eurostat - Urban-rural Europe- labour market.pdf"
}

build_configurations = [
    {
        "embedding_model": embedding_model,
        "fixed_chunk_size": chunk_size
    }
    for embedding_model in SUPPORTED_EMBEDDING_MODELS
    for chunk_size in [150, 300]
]

retrieval_configurations = [
    {
        "num_context_chunks": num_chunks,
        "range_neighbor_chunks": neighbor_range
    }
    for num_chunks in [1,2,4]
    for neighbor_range in [0,1,3]
]


def GetPDFPath(database_name):
    if database_name not in DATABASE_PATHS:
        raise ValueError(
            f"No PDF path configured for database '{database_name}'."
        )

    return DATABASE_PATHS[database_name]

def ParseArgs(args):
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the csv document."
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="RAG_test_results.csv",
        help="Path where evaluation results will be stored."
    )
        
    return parser.parse_args(args)
    
def ParseCSV(csv_path):
    path = Path(csv_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"CSV file does not exist: {csv_path}"
        )

    csv_rows = []

    with path.open(
        mode="r",
        encoding="utf-8-sig",
        newline=""
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError("CSV file does not contain a header row.")

        for row in reader:
            csv_rows.append(row)

    return csv_rows

def StoreResultsCSV(results, output_path):
    if not results:
        raise ValueError("No evaluation results to store.")

    path = Path(output_path)

    with path.open(
        mode="w",
        encoding="utf-8-sig",
        newline=""
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)


def Evaluate(
    csv_rows,
    database_name,
    embedding_model,
    fixed_chunk_size,
    num_context_chunks,
    range_neighbor_chunks,
    construction_elapsed_time
):
    results = []

    for test in csv_rows:
        start_time = time.perf_counter()

        retrieved_context, answer = GetRAGsAnswer(
            database_name,
            embedding_model,
            test["question"],
            num_context_chunks,
            range_neighbor_chunks
        )

        elapsed_time = time.perf_counter() - start_time

        result = {
            "test_id": test["test_id"],
            "database_name": database_name,

            "embedding_model": embedding_model,
            "fixed_chunk_size": fixed_chunk_size,
            "num_context_chunks": num_context_chunks,
            "range_neighbor_chunks": range_neighbor_chunks,

            "question": test["question"],
            "answerable": test["answerable"],

            "expected_context": test["expected_context"],
            "retrieved_context": retrieved_context,

            "expected_answer": test["expected_answer"],
            "answer": answer,

            "retrival_time": elapsed_time,
            "construction_time": construction_elapsed_time
        }

        results.append(result)

    return results

def BuildAndEvaluate(csv_rows):
    all_results = []    

    valid_tests = [
        test
        for test in csv_rows
        if test["database_name"] in DATABASE_PATHS
    ]

    total_runs = (
        len(valid_tests)
        * len(build_configurations)
        * len(retrieval_configurations)
    )
    i = 0

    for build_config in build_configurations:

        embedding_model = build_config["embedding_model"]
        fixed_chunk_size = build_config["fixed_chunk_size"]

        for database_name in DATABASE_PATHS:

            database_tests = [
                test
                for test in csv_rows
                if test["database_name"] == database_name
            ]

            if not database_tests:
                continue

            pdf_path = GetPDFPath(database_name)

            # We only rebuild for embedding_model * fixed_chunk_size variations
            start_time = time.perf_counter()
            ConstructRAG(
                pdf_path,
                fixed_chunk_size,
                True,
                database_name,
                embedding_model
            )
            construction_elapsed_time = time.perf_counter() - start_time

            for retrieval_config in retrieval_configurations:

                num_context_chunks = retrieval_config["num_context_chunks"]
                range_neighbor_chunks = retrieval_config["range_neighbor_chunks"]

                results = Evaluate(
                    database_tests,
                    database_name,
                    embedding_model,
                    fixed_chunk_size,
                    num_context_chunks,
                    range_neighbor_chunks,
                    construction_elapsed_time
                )
                i+=len(database_tests)
                print(f"Completed {i}/{total_runs} so far")
                all_results.extend(results)
                
    return all_results


def main():    
    args = ParseArgs(sys.argv[1:])
    
    csv_path = args.input_path
    output_path = args.output_path
    
    try:
        csv_rows = ParseCSV(csv_path)
    except Exception as e:
        print(f"Failed reading CSV: {e}")
        return 1
    
    try:
        results = BuildAndEvaluate(csv_rows)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 1
    
    print(f"Completed {len(results)} evaluation runs.")
        
    try:
        StoreResultsCSV(results, output_path)
    except Exception as e:
        print(f"Failed storing results: {e}")
        return 1
    
    print(f"Results stored in: {output_path}")
        

if __name__ == "__main__":
    main()
# small-local-rag-prototype
Mali prototip lokalnog rag sustava koji koristi ChromaDB i LM Studio.

## Architecture
PDF

 ↓
Text extraction

 ↓
Paragraph + sentence-aware chunking

 ↓
all-MiniLM-L6-v2 embeddings

 ↓
ChromaDB

 ↓
Semantic retrieval

 ↓
Neighbor chunk expansion

 ↓
Gemma 3 270M through LM Studio

Answer


## Requirements
- Python
- ChromaDB
- openai
- pypdf
- sentence-transformer
- LM studio
- spaCy

## Installation
### Python dependencies
Install the required Python packages:
```bash
pip install chromadb openai pypdf sentence-transformers spacy
```

The prototype uses spaCy's English model for sentence segmentation. Install it with:
```bash
python -m spacy download en_core_web_sm
```

### LM Studio
Install [LM Studio](https://lmstudio.ai/) and download a local language model.
The current prototype is configured to use `gemma-3-270m-it-qat`

Load the model in LM Studio and start the local API server.
The prototype expects the OpenAI-compatible API to be available at `http://localhost:1234/v1`

After the Python dependencies, spaCy model, and LM Studio are set up, the RAG system can be run using the commands described in the Usage section.


## Usage

The RAG prototype is used in three main steps:

1. Build a ChromaDB collection from a PDF document.
2. Retrieve relevant context from the collection.
3. Send the retrieved context and question to a local LLM through LM Studio.

### 1. Build the database

Run `RAG_constructor.py` to extract text from a PDF, split it into chunks, generate embeddings, and store them in a persistent ChromaDB collection.

Example:

```bash
py RAG_constructor.py --path "example_PDFs\Wuthering heights.pdf" --database_name wuthering_heights --new_embeddings --fixed_chunk_size 250
```

Arguments:

* `--path` - path to the PDF document.
* `--database_name` - name of the ChromaDB collection. The same name must be used when retrieving context or asking questions.
* `--fixed_chunk_size` - target minimum chunk size in characters. The default value is `250`.
* `--new_embeddings` - deletes the existing collection with the same name before rebuilding it.

The resulting database is stored in a directory named `chroma_database_<database_name>`

For example `chroma_database_wuthering_heights`

NOTICE: The current text-processing pipeline is intended for text-heavy PDF documents. Text is first separated into paragraphs and then segmented into sentences using spaCy's `en_core_web_sm` model.

NOTICE: Because sentence segmentation currently uses `en_core_web_sm`, documents should primarily contain English text unless the NLP model is changed.

### 2. Retrieve context

`Context_provider.py` can be used independently to inspect which parts of the document are retrieved for a question.

Example:

```bash
py Context_provider.py --query "How long did Cathy stay at the Thrushcross Grange?" --database_name wuthering_heights --num_context_chunks 2
```

Arguments:

* `--query` - question used for semantic retrieval.
* `--database_name` - name of the previously created ChromaDB collection.
* `--num_context_chunks` - number of chunks retrieved directly through semantic search.

For every semantically retrieved chunk, the system also attempts to retrieve its immediately previous and next chunks. This provides additional surrounding context before the text is sent to the language model.

The expanded chunks are reordered according to their original position in the document before being combined into the final context.

### 3. Ask the RAG system a question

Before running `Prompt_sender.py`, start LM Studio and load the model expected by the script.

The current prototype uses `gemma-3-270m-it-qat`

LM Studio must expose its OpenAI-compatible local API at `http://localhost:1234/v1`

Once the model and local server are running, ask a question with:

```bash
py Prompt_sender.py --query "How long did Cathy stay at the Thrushcross Grange?" --database_name wuthering_heights --num_context_chunks 2
```

The script performs semantic retrieval, expands the retrieved chunks with neighboring chunks, and sends the resulting context together with the question to the local LLM.

The model is instructed to answer using only the retrieved context. If the retrieved context does not explicitly contain the answer, it should respond `The answer is not stated in the retrieved context.`

The command prints:

* the selected LLM,
* the original question,
* the retrieved context,
* and the generated answer.

### Example workflow

A complete workflow using *Wuthering Heights* is:

```bash
# Build the collection
py RAG_constructor.py --path "example_PDFs\Wuthering heights.pdf" --database_name wuthering_heights --new_embeddings --fixed_chunk_size 250

# Inspect retrieved context
py Context_provider.py --query "How long did Cathy stay at the Thrushcross Grange?" --database_name wuthering_heights --num_context_chunks 2

# Generate the final answer
py Prompt_sender.py --query "How long did Cathy stay at the Thrushcross Grange?" --database_name wuthering_heights --num_context_chunks 2
```

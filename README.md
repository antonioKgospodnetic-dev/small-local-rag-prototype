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
 ↓
Answer


## Requirements
- Python
- ChromaDB
- openai
- pypdf
- sentence-transformer
- LM studio
- spaCy

## Instalations
python -m spacy download en_core_web_sm
...

## Building the database
- where does it hapeen
- how does it work
- how do I affect it
- what it requires
- warnings

The database is built using python commands in RAG_constructor.py. 
The script is run with a command such as:
''' 
    py RAG_constructor.py --path "example_PDFs\Wuthering heights.pdf" --database_name wuthering_heights --new_embedding --fixed_chunk_size 250
'''

After parsing the arguments, it uses pypdfs PdfReader to get the text from the pdf.
NOTICE: It is designed to handle text heavy pdfs, ones without images or graphs

The retrieved text is then sent to the chunking algoritm.
Its designed to seperate the tekst into paragraphs, and then segment into sentences using spaCy nlp
NOTICE: The NLP used is "en_core_web_sm", meaning different pdfs must align with this model or change which one is used.

Chunks are filled with sentences until they are at least over the desired minimal "fixe_chunk_size" (character amount)


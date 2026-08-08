import sys
from pypdf import PdfReader
import re
import spacy
import chromadb
import chromadb.utils.embedding_functions
  
def ParseArgs(args):
    # cmd_params = {"--path","--fixed_chunk_size","--new_embeddings"}
    pdf_path, fixed_chunk_size, new_embeddings = None,None,None
    for i_arg in range(0, len(args)):
        if (args[i_arg] == "--path"):
            i_arg = i_arg + 1
            pdf_path = args[i_arg]
        elif (args[i_arg] == "--fixed_chunk_size"):
            i_arg = i_arg + 1                
            fixed_chunk_size = int(args[i_arg])
        elif (args[i_arg] == "--new_embeddings"):
            i_arg = i_arg + 1                
            new_embeddings = bool(args[i_arg])
            
    # default
    if(fixed_chunk_size == None):
        fixed_chunk_size = 250
    if(new_embeddings == None):
        new_embeddings = False        
        
    return pdf_path, fixed_chunk_size, new_embeddings
  
def ParsePDF(path):
    reader = PdfReader(path)
    pdf_pages_text = []
    for page in reader.pages:
        extract = page.extract_text() or ""
        pdf_pages_text.append(extract)
        
    pdf_text = "\n".join(pdf_pages_text)
    
    return pdf_text
  
def FixedSizeChunking(text, chunk_size):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[(i):(chunk_size+i)])
    return chunks
  
def ParagraphedSentenceAwareFixedSizeChunking(text, target_chunk_size):
    FREE_SPACE = 50
    CONNECTOR = " "
    
    paragraphs = re.split(r"\n\s*\n+", text)

    paragraphs = [
        re.sub(r"\s*\n\s*", " ", paragraph).strip()
        for paragraph in paragraphs
    ]
    
    paragraphs = [
        paragraph for paragraph in paragraphs
        if paragraph
    ]
    
    chunks = []    
    # Using spaCy for sentence segmentation
    # This means the chunking algorith is language dependant
    nlp = spacy.load("en_core_web_sm")
    
    # If previous paragraph sentance doesnt reach minimum character chunk size, this flag is False
    prev_chunk_full_flag = True
    
    for paragraph in paragraphs:
        if(prev_chunk_full_flag):
            current_chunk = ""
        
        doc = nlp(paragraph)
        sentences = list(doc.sents)
        
        for i_sentence in range(len(sentences)):
            sentence = str(sentences[i_sentence])
            
            if ((current_chunk == "") or (prev_chunk_full_flag == False)):
                current_chunk += sentence
                prev_chunk_full_flag = True
            else:
                current_chunk = current_chunk + CONNECTOR + sentence
                
            if(i_sentence + 1 < len(sentences)):
                if((len(current_chunk) + len(CONNECTOR) + len(sentences[i_sentence + 1].text)) > (target_chunk_size + FREE_SPACE)):
                    chunks.append(current_chunk)
                    current_chunk = ""
            
        if (len(current_chunk) < target_chunk_size):
            prev_chunk_full_flag = False
            current_chunk += "\n\n"
        else:
            chunks.append(current_chunk)
    
    if(prev_chunk_full_flag == False):
        chunks.append(current_chunk)
    
    return chunks
   
      
#py RAG_constructor.py --path PDFs_for_Parsing\"Wuthering heights.pdf" --new_embeddings True --fixed_chunk_size 250
def main():
    args = sys.argv[1:]
    
    pdf_path, fixed_chunk_size, new_embeddings = ParseArgs(args)
    
    if ((fixed_chunk_size < 1) or (fixed_chunk_size>2500)):
        print("err: Incorrect fixed chunk size")
        exit(1)

    text = ParsePDF(pdf_path)
        
    chunks = ParagraphedSentenceAwareFixedSizeChunking(text, fixed_chunk_size)
    
    # Embedding model
    embedding_function = (
        chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )
    
    client = chromadb.PersistentClient(
        path="./chroma_database"
    )

    # If flag is true, cleanly rebuild
    if (new_embeddings):
        try:
            client.delete_collection(name="wuthering_heights")
            print("Succesfully deleted existing collection")
        except Exception:
            print("Failed deleting existing collection")
            pass

    collection = client.get_or_create_collection(
        name="wuthering_heights",
        embedding_function=embedding_function
    )

    # Each chunk needs an ID ( it will be provided with a format "chunk_X" )
    chunk_ids = []
    # Each chunk will be provided metadata about its neighboring chunks
    chunk_metadatas = []

    for i in range(len(chunks)):
        chunk_id = "chunk_" + str(i)
        chunk_ids.append(chunk_id)

        previous_chunk_id = "chunk_" + str(i - 1) if i > 0 else ""
        next_chunk_id = "chunk_" + str(i + 1) if i < len(chunks) - 1 else ""

        chunk_metadatas.append({
            "chunk_index": i,
            "previous_chunk_id": previous_chunk_id,
            "next_chunk_id": next_chunk_id
        })

    collection.upsert(
        ids=chunk_ids,
        documents=chunks,
        metadatas=chunk_metadatas
    )

    # Number of chunks in collection must match provided chunks
    print("Number of chunks:", len(chunks))
    print("Number of chunks stored in ChromaDB:", collection.count())
        

if __name__ == "__main__":
    main()
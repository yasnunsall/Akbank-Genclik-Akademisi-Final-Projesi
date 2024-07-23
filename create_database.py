import os
import shutil
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from pypdf import PdfReader
from chromadb.config import DEFAULT_TENANT,  DEFAULT_DATABASE, Settings
from chromadb import Client, PersistentClient
from chromadb.utils import embedding_functions
import textwrap
from IPython.display import display
from IPython.display import Markdown
import warnings
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore")

sentence_transformer_model = "paraphrase-multilingual-MiniLM-L12-v2"
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name = sentence_transformer_model)

def to_markdown(text):
    text = text.replace("•", " *")
    return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))

chromaDB_path = "ChromaDBData"

def create_chroma_client(collection_name, embedding_function, chromaDB_path):
    if chromaDB_path is not None:
        chroma_client = PersistentClient(path = chromaDB_path,
                                         settings=Settings(),
                                         tenant=DEFAULT_TENANT,
                                         database=DEFAULT_DATABASE)
    else:
        chroma_client = Client()

    chroma_collection = chroma_client.get_or_create_collection(
        collection_name,
        embedding_function=embedding_function
    )
    return chroma_client, chroma_collection

def convert_PDF_Text(pdf_path):
    reader = PdfReader(pdf_path)
    pdf_texts = [p.extract_text().strip() for p in reader.pages]
    pdf_texts = [text for text in pdf_texts if text]
    print("Document: ", pdf_path, " chunk size: ", len(pdf_texts))
    return pdf_texts

def convert_Page_ChunkinChar(pdf_texts, chunk_size = 1000, chunk_overlap = 100):
    character_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", ". ", " ", ""],
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))
    print(f"\nToplam chunk sayısı: {len(character_split_texts)}")
    return character_split_texts

def convert_Chunk_Token(text_chunksinChar, sentence_transformers_model, chunk_overlap=100, tokens_per_chunk=128):
    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap = chunk_overlap,
        model_name = sentence_transformers_model,
        tokens_per_chunk = tokens_per_chunk)
    text_chunksinTokens = []
    for text in text_chunksinChar:
        text_chunksinTokens += token_splitter.split_text(text)
    return text_chunksinTokens

def add_meta_data(text_chunksinTokens, title, category, initial_id):
  ids = [str(i+initial_id) for i in range(len(text_chunksinTokens))]
  metadata = { 'document': title, 'category': category }
  metadatas = [ metadata for i in range(len(text_chunksinTokens))]
  return ids, metadatas

def add_document_to_collection(ids, metadatas, text_chunksinTokens, chroma_collection):
  chroma_collection.add(ids=ids, metadatas= metadatas, documents=text_chunksinTokens)
  return chroma_collection

def load_pdf_to_ChromaDB(collection_name,sentence_transformer_model, chromaDB_path, file_path):
  category= "Banka"
  embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(sentence_transformer_model)
  chroma_client, chroma_collection = create_chroma_client(collection_name, embedding_function, chromaDB_path)
  current_id = chroma_collection.count()
  file_name = file_path.strip("/")[-1]
  pdf_texts = convert_PDF_Text(file_path)
  text_chunksinChar = convert_Page_ChunkinChar(pdf_texts)
  text_chunksinTokens = convert_Chunk_Token(text_chunksinChar,sentence_transformer_model)
  ids, metadatas = add_meta_data(text_chunksinTokens,file_name,category, current_id)
  chroma_collection = add_document_to_collection(ids, metadatas, text_chunksinTokens, chroma_collection)
  return  chroma_client, chroma_collection

file_path = "rag_data/akbank.pdf"

if __name__ == "__main__":
    chroma_client, chroma_collection = load_pdf_to_ChromaDB("Akbank", sentence_transformer_model, chromaDB_path, file_path)
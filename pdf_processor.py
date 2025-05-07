"""
PDF Processor Module for Vector DB Creation without Difficulty Metadata
"""

import os
import shutil
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from fastapi import FastAPI, Form, HTTPException
import uvicorn

app = FastAPI()

# API key configuration
os.environ["GOOGLE_API_KEY"] = ""
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Vector DB folder
VECTOR_DB_FOLDER = 'static/vector_db'
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

def count_pdf_pages(pdf_path):
    try:
        pdf = PdfReader(pdf_path)
        return len(pdf.pages)
    except Exception as e:
        print(f"Error counting PDF pages: {e}")
        return None

def split_pdf_to_documents(file_path, chapter_title="Default Chapter"):
    """Read PDF and split into documents without difficulty metadata."""
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        text = "".join([p.page_content for p in pages])

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)

        docs = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    "chapter": chapter_title
                }
            )
            docs.append(doc)

        return docs
    except Exception as e:
        print(f"Error splitting PDF: {e}")
        return []

def save_to_faiss(documents, index_path):
    try:
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
            print(f"⚠️ Existing vector DB at {index_path} has been removed.")
        os.makedirs(index_path, exist_ok=True)
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(index_path)
        print(f"✅ Vector DB saved to: {index_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving Vector DB: {e}")
        return False

def process_pdf_and_create_vectordb(pdf_path, chapter_title="Default Chapter"):
    pdf_filename = os.path.basename(pdf_path)
    vector_db_path = os.path.join(VECTOR_DB_FOLDER, pdf_filename.replace('.pdf', ''))

    print(f"📄 Reading and splitting PDF: {pdf_path}")
    documents = split_pdf_to_documents(pdf_path, chapter_title)

    if not documents:
        return None, "Failed to process PDF documents"

    print(f"📊 Total chunks: {len(documents)}")
    print("💾 Saving to vector DB...")
    success = save_to_faiss(documents, vector_db_path)

    if success:
        return vector_db_path, None
    else:
        return None, "Failed to save to vector database"

@app.post("/process")
async def process_file(pdf_filename: str = Form(...), chapter_title: str = Form("Default Chapter")):
    if not os.path.exists(pdf_filename):
        raise HTTPException(status_code=404, detail=f"PDF file not found: {pdf_filename}")

    vector_db_path, error = process_pdf_and_create_vectordb(pdf_filename, chapter_title)

    if error:
        raise HTTPException(status_code=500, detail=error)

    return {
        "message": "Vector DB created successfully",
        "vector_db_path": vector_db_path
    }

if __name__ == "__main__":
    uvicorn.run("pdf_processor:app", host='0.0.0.0', port=8001, reload=True)

from pathlib import Path
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

class DocumentLoader:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_documents(self, file_path: str) -> List:
        """Load documents from PDF or TXT files."""
        path = Path(file_path)
        
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(file_path)
        elif path.suffix.lower() == ".txt":
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def load_from_folder(self, folder_path: str) -> List:
        """Load all documents from a folder."""
        all_docs = []
        folder = Path(folder_path)
        
        for file_path in folder.glob("**/*.{pdf,txt}"):
            try:
                docs = self.load_documents(str(file_path))
                all_docs.extend(docs)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return all_docs
import os
from pathlib import Path
from dataclasses import dataclass
import torch
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from typing import Dict, Any, List
from langchain.text_splitter import RecursiveCharacterTextSplitter

from tqdm import tqdm

@dataclass
class RAGConfig:
    """RAG system configuration"""
    # Azure OpenAI settings
    azure_api_key: str
    azure_api_base: str
    azure_deployment_name: str
    azure_api_version: str = "2023-05-15"
    temperature: float = 0.1

    # Document processing settings
    chunk_size: int = 1000
    chunk_overlap: int = 100
    top_k: int = 50

    # Paths
    docs_dir: str = "docs"
    persist_dir: str = "db"
    
    # Embedding model
    embedding_model_name: str = "all-MiniLM-L6-v2"
    force_reload: bool = False

class RAGSystem:
    def __init__(self, config: RAGConfig):
        self.config = config
        print(f"\nInitializing RAG system...")
        print(f"Document directory: {self.config.docs_dir}")
        print(f"Vector store directory: {self.config.persist_dir}")
        self.setup_models()
        self.setup_vector_store()
        self.setup_rag_chain()

    def setup_models(self):
        """Initialize embedding and language models"""
        print("Setting up models...")
        # Setup embedding model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Setup Azure OpenAI
        self.llm = AzureChatOpenAI(
            openai_api_key=self.config.azure_api_key,
            azure_endpoint=self.config.azure_api_base,
            openai_api_version=self.config.azure_api_version,
            deployment_name=self.config.azure_deployment_name,
            temperature=self.config.temperature
        )

    def load_pdfs(self) -> List:
        """Load all PDFs recursively from the directory and its subdirectories"""
        # Check if directory exists
        if not os.path.exists(self.config.docs_dir):
            raise ValueError(f"Directory does not exist: {self.config.docs_dir}")
        
        # Recursively find all PDF files
        pdf_files = []
        for root, _, files in os.walk(self.config.docs_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    full_path = os.path.join(root, file)
                    pdf_files.append(full_path)
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.config.docs_dir} or its subdirectories")
        
        print(f"Found {len(pdf_files)} PDF files")
        
        # Load each PDF
        documents = []
        for pdf_path in pdf_files:
            try:
                relative_path = os.path.relpath(pdf_path, self.config.docs_dir)
                print(f"Loading {relative_path}...")
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                # Add source information to each document
                for doc in docs:
                    doc.metadata['source'] = relative_path
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {pdf_path}: {str(e)}")
                continue
        
        print(f"Successfully loaded {len(documents)} pages from {len(pdf_files)} PDFs")
        return documents

    def setup_vector_store(self):
        """Setup or load vector store"""
        persist_path = Path(self.config.persist_dir)

        if not persist_path.exists() or self.config.force_reload:
            print("Creating new vector store...")
            self._create_new_vector_store()
        else:
            print("Loading existing vector store...")
            self._load_existing_vector_store()
    

    def _create_new_vector_store(self):
        """Create new vector store from documents"""
        try:
            # Load all PDFs
            documents = self.load_pdfs()
            
            if not documents:
                raise ValueError(f"No documents were loaded from {self.config.docs_dir}")
            
            # Create text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            
            # Split documents
            print("Splitting documents...")
            split_documents = text_splitter.split_documents(documents)
            total_chunks = len(split_documents)
            print(f"Total chunks to process: {total_chunks}")
            
            # Create and persist vector store in batches
            print("Creating vector store...")
            batch_size = 40000  # Keep below 41666 limit
            
            # Process first batch
            first_batch = split_documents[:batch_size]
            print("Processing first batch...")
            self.vector_store = Chroma.from_documents(
                documents=first_batch,
                embedding=self.embeddings,
                persist_directory=self.config.persist_dir
            )
            
            # Process remaining batches
            remaining_docs = split_documents[batch_size:]
            if remaining_docs:
                num_batches = (len(remaining_docs) + batch_size - 1) // batch_size
                print(f"Processing {num_batches} additional batches...")
                
                with tqdm(total=len(remaining_docs), desc="Processing documents") as pbar:
                    for i in range(0, len(remaining_docs), batch_size):
                        batch = remaining_docs[i:i + batch_size]
                        self.vector_store.add_documents(batch)
                        pbar.update(len(batch))
            
            # Persist the final vector store
            print("Persisting vector store...")
            self.vector_store.persist()
            print(f"Successfully created vector store with {total_chunks} chunks")
            
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            raise


    def _load_existing_vector_store(self):
        """Load existing vector store"""
        self.vector_store = Chroma(
            persist_directory=self.config.persist_dir,
            embedding_function=self.embeddings
        )

    def setup_rag_chain(self):
        """Setup the RAG processing chain"""
        # Define prompt template
        template = """Answer the question based on the following context. 
        Try your best to answer the question based on the context. Try and answer with whatever relevenat information you find in the context. If there is absolutely no relevant infomration in the context, then say "I don't have enough information to answer this question."

        Context: {context}

        Question: {question}

        Answer: """

        prompt = ChatPromptTemplate.from_template(template)

        # Setup retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.top_k}
        )

        # Create RAG chain
        self.chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
                "history": itemgetter("history")
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str, history = "") -> str:
        """Process a query through the RAG system"""
        try:
            return self.chain.invoke({"question": question,
                                      "history": history})
        except Exception as e:
            return f"Error processing query: {str(e)}"
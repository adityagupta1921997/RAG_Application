from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from prompts import RAG_PROMPT_TEMPLATE


class RAG:

    def __init__(self, pdf_path) -> None:
        self.pdf_path = pdf_path

    def extract_documents_from_pdf(self):
        """loads and split given pdf to document chunks"""
        loader = PDFPlumberLoader(self.pdf_path)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(pages)
        return documents

    def load_vector_store(self, documents):
        """creates vector store and add all the documents to vector store"""
        client = QdrantClient(path="langchain_qdrant")

        client.create_collection(
            collection_name="demo_collection",
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="demo_collection",
            embedding=embed_model,
        )
        vector_store.add_documents(documents=documents)
        return vector_store

    def get_retriever(self, vector_store):
        """creates retriver from vector store"""
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        # can use re-ranking to increase retrieval accuracy
        # compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
        # compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
        #                                                         base_retriever=retriever)
        return retriever

    def load_llm(self):
        """returns llm model to be used answer generation"""
        groq_api_key = "<GROQ_API_KEY>"
        llm=ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
        return llm

    def get_qa_chain(self):
        """returns qa chain"""
        documents = self.extract_documents_from_pdf()
        vector_store = self.load_vector_store(documents)
        retriever = self.get_retriever(vector_store)
        llm = self.load_llm()
        prompt = PromptTemplate(template=RAG_PROMPT_TEMPLATE, input_variables=["context", "question"])
        qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "verbose": False},
        )
        return qa_chain
    
    def generate_answer(self, query):
        """returns answer for given user query"""
        qa_chain = self.get_qa_chain()
        response = qa_chain.invoke(query)
        return response['result']
    
if __name__ == "__main__":
    pdf_path = "sample.pdf"
    rag = RAG(pdf_path)
    query = "What is the warranty period of this car?"
    rag.generate_answer(query)
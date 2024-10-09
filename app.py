import streamlit as st
from rag import RAG

pdf_path = "sample.pdf"
rag = RAG(pdf_path)

def main():
    st.set_page_config(page_title="Chatbot Assistant", layout="centered")

    st.title("AI-Powered Question Answering System")
    
    st.write("""
    Welcome to the AI-powered question-answering application! 
    Just type in your question below, and let the AI provide an accurate response.
    """)

    # Create a text input for the question
    question = st.text_input("Enter your question:", "")

    # Answering section
    if question:
        with st.spinner("Fetching answer..."):
            answer = rag.generate_answer(question)
            st.write(f"**Answer:** {answer}")

# Main entry point
if __name__ == "__main__":
    main()

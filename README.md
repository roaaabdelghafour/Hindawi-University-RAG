# 📚 Hindawi University RAG Assistant

🌐 **Live Demo:** https://hindawi-university-rag-mm9vded9tstckf8oxehkps.streamlit.app/

A Retrieval-Augmented Generation (RAG) application built with Streamlit...

A Retrieval-Augmented Generation (RAG) application built with Streamlit that allows users to ask questions about Hindawi University information stored in a PDF document.

## 🚀 Project Overview

This project uses RAG to retrieve relevant information from a university information PDF and generate an answer based only on the retrieved context.

The application combines:

- PDF text extraction
- Text chunking
- Sentence embeddings
- FAISS vector search
- FLAN-T5 for answer generation
- Streamlit for the user interface

## 🛠️ Technologies Used

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- Hugging Face Transformers
- FLAN-T5
- PyTorch

## 🔄 How It Works

The application follows these steps:

1. Reads the PDF document.
2. Extracts the text from the PDF.
3. Splits the text into smaller chunks.
4. Converts the chunks into embeddings.
5. Stores the embeddings in a FAISS index.
6. Converts the user's question into an embedding.
7. Retrieves the most relevant chunks.
8. Sends the retrieved context and question to FLAN-T5.
9. Displays the generated answer in the Streamlit interface.

## 📁 Project Structure

```text
RAG_Streamlit/
│
├── app.py
├── Tips Hindawi University Info.pdf
├── requirements.txt
└── README.md

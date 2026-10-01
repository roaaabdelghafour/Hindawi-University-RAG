import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import torch


# -----------------------------
# Streamlit Page Settings
# -----------------------------
st.set_page_config(
    page_title="Hindawi University RAG",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Hindawi University RAG Assistant")
st.write("Ask questions about Tips Hindawi University.")


# -----------------------------
# PDF Path
# -----------------------------
pdf_path = "Tips Hindawi University Info.pdf"


# -----------------------------
# Read PDF
# -----------------------------
reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    text += page.extract_text() + "\n"


# -----------------------------
# Split Text Into Chunks
# -----------------------------
chunk_size = 500
overlap = 100

chunks = []

start = 0

while start < len(text):
    end = start + chunk_size
    chunks.append(text[start:end])
    start += chunk_size - overlap


# -----------------------------
# Create Embeddings
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()

embeddings = embedding_model.encode(
    chunks,
    convert_to_numpy=True
)


# -----------------------------
# Create FAISS Index
# -----------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


# -----------------------------
# Retrieve Relevant Chunks
# -----------------------------
def retrieve(question, k=3):

    # Convert the question into an embedding
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    # Search for the most relevant chunks
    distances, indices = index.search(
        question_embedding,
        k
    )

    # Get the retrieved chunks
    retrieved_chunks = [chunks[i] for i in indices[0]]

    return retrieved_chunks
# -----------------------------
# Load FLAN-T5 Model
# -----------------------------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_generation_model():

    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map=None
    )

    model = model.to("cpu")

    return tokenizer, model


tokenizer, model = load_generation_model()
# -----------------------------
# Generate Answer
# -----------------------------
def generate_answer(prompt):

    # Convert the prompt into tokens
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Make sure inputs are on CPU
    inputs = {
        key: value.to("cpu")
        for key, value in inputs.items()
    }

    # Generate the answer
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100
        )

    # Convert tokens back to text
    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer
# -----------------------------
# RAG Answer
# -----------------------------
def rag_answer(question):

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(question, k=3)

    # Combine chunks into one context
    context = "\n".join(retrieved_chunks)

    # Create the prompt
    prompt = f"""
Answer the question using only the information in the context.

Context:
{context}

Question:
{question}

Answer:
"""

    # Generate the answer
    answer = generate_answer(prompt)

    return answer
# -----------------------------
# User Question
# -----------------------------
st.subheader("Ask a Question")

question = st.text_input(
    "Enter your question:",
    placeholder="e.g. Where is Hindawi University located?"
)

if st.button("Get Answer", type="primary"):

    if question.strip():

        with st.spinner("Searching the document and generating the answer..."):

            # Retrieve relevant chunks
            retrieved_chunks = retrieve(question, k=3)

            # Combine chunks
            context = "\n".join(retrieved_chunks)

            # Create prompt
            prompt = f"""
Answer the question using only the information in the context.

Context:
{context}

Question:
{question}

Answer:
"""

            # Generate answer
            answer = generate_answer(prompt)

        # -----------------------------
        # Display Answer
        # -----------------------------
        st.subheader("Answer")

        st.info(answer)

        # -----------------------------
        # Retrieved Context
        # -----------------------------
        with st.expander("View Retrieved Context"):

            for i, chunk in enumerate(retrieved_chunks, 1):

                st.markdown(f"**Chunk {i}**")

                st.write(chunk)

    else:

        st.warning("Please enter a question.")
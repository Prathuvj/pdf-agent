import chainlit as cl
import google.generativeai as genai
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

# Load environment variables
load_dotenv()

# Set up Gemini API
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
genai.configure(api_key=api_key)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

# Function to process the uploaded PDF
def process_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store
def create_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

# Function to get response from Gemini
def get_gemini_response(query, vector_store):
    docs = vector_store.similarity_search(query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    # Use the selected model
    model = genai.GenerativeModel(selected_model)
    response = model.generate_content(prompt)
    return response.text

# Function to create a PDF
def create_pdf(query, response):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Add title
    story.append(Paragraph("Document Q&A Response", styles['Title']))
    story.append(Spacer(1, 12))

    # Add query
    story.append(Paragraph("Question:", styles['Heading2']))
    story.append(Paragraph(query, styles['BodyText']))
    story.append(Spacer(1, 12))

    # Add response
    story.append(Paragraph("Answer:", styles['Heading2']))
    story.append(Paragraph(response, styles['BodyText']))

    doc.build(story)
    buffer.seek(0)
    return buffer

# Chainlit UI Implementation

vector_store = None

from chainlit.input_widget import Select

@cl.on_chat_start
async def start():
    global vector_store, selected_model
    # Model selection
    settings = await cl.ChatSettings([
        Select(
            id="Model",
            label="Gemini Model",
            values=["gemini-2.5-flash-preview-05-20", "gemini-2.5-pro-preview-05-06"],
            initial_index=0,
        )
    ]).send()
    selected_model = settings["Model"]

    # Multi-file upload
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload up to 10 PDF documents (max 250 MB each) to begin!",
            accept=["application/pdf"],
            max_size_mb=250,
            max_files=10
        ).send()
    try:
        all_text = ""
        for file in files:
            with open(file.path, "rb") as f:
                text = process_pdf(f)
                if not text.strip():
                    await cl.Message(content=f"No text could be extracted from {file.name}. Skipping this file.").send()
                    continue
                all_text += text + "\n"
        if not all_text.strip():
            await cl.Message(content="No text could be extracted from any PDF. Please try different documents.").send()
            return
        text_chunks = split_text(all_text)
        vector_store = create_vector_store(text_chunks)
        await cl.Message(
            content="All documents processed successfully! You can now ask questions about the combined content."
        ).send()
    except Exception as e:
        await cl.Message(content=f"An error occurred while processing the PDFs: {e}").send()
        raise


@cl.on_message
async def on_message(message: cl.Message):
    global vector_store
    if vector_store is None:
        await cl.Message(content="Please upload a PDF document first.").send()
        return
    query = message.content.strip()
    if not query:
        await cl.Message(content="Please enter a question.").send()
        return
    response = get_gemini_response(query, vector_store)
    await cl.Message(content=f"**Answer:**\n{response}").send()
    # Offer PDF download
    pdf_buffer = create_pdf(query, response)
    await cl.File(name="response.pdf", content=pdf_buffer.getvalue(), display_name="Download Response as PDF").send()
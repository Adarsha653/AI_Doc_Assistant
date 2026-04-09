from dotenv import load_dotenv
load_dotenv()
import chainlit as cl
import os
from src.chat import ChatInterface
from src.document_loader import DocumentLoader

chat = None

@cl.on_chat_start
async def start():
    global chat
    chat = ChatInterface()
    
    # Enable file uploads
    files = await cl.AskFileMessage(
        content="Upload a PDF or TXT file to get started!",
        accept=["application/pdf", "text/plain"],
        max_size_mb=20
    ).send()
    
    if files:
        for file in files:
            await cl.Message(content=f"Processing {file.name}...").send()
            try:
                loader = DocumentLoader()
                documents = loader.load_documents(file.path)
                chat.qa_system.build_vectorstore(documents)
                await cl.Message(content=f"✅ Successfully loaded {file.name}! Ask me anything about it.").send()
            except Exception as e:
                await cl.Message(content=f"❌ Error: {str(e)}").send()

@cl.on_message
async def main(message: cl.Message):
    try:
        response = chat.process_query(message.content)
        await cl.Message(content=response).send()
    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()
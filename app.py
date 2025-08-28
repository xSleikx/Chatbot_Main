import gradio as gr
import ollama
import os
import re
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

import pdfRAG
import audioRAG

# Load Groq API key from environment variable
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# Initialize conversation history with system message
history = [{"role": "system", "content": "Welcome to the Audio & PDF RAG Chatbot!"}]

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function to reset the conversation history
def reset_h():
    global history
    history = [{"role": "system", "content": "Welcome to the Audio & PDF RAG Chatbot!"}]

# Function to handle file uploads
def upload_file(file, message, chunk_size_slider, overlap_slider, model_selector):
    global history
    try:
        if not message.strip():
            return "", history, "", []
        
        if file.endswith(('.wav', '.mp3')):
            retrieved_docs = audioRAG.load_and_retrieve_audio_docs(file, message, chunk_size_slider, overlap_slider)
        else:
            retrieved_docs = pdfRAG.load_and_retrieve_docs(file, message, chunk_size_slider, overlap_slider)
        
        formatted_context = format_docs(retrieved_docs)
        formatted_prompt = f"Question: {message}\n\nContext: {formatted_context}"

        # Prepare messages including the current user prompt
        messages = history.copy()
        messages.append({"role": "user", "content": formatted_prompt})

        # Check if the selected model is a Groq model
        groq_models = ["llama3-70b-8192", "llama-3.3-70b-specdec", "deepseek-r1-distill-llama-70b"]
        if model_selector in groq_models:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model_selector,
            )
            response = chat_completion.choices[0].message.content
        else:
            chat = ollama.chat(model=model_selector, messages=messages)
            response = chat['message']['content']
        # deepseek format response
        if "deepseek" in model_selector.lower():
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()   
        # Update history with user's question and the response
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return "", history, "", []
    except Exception as e:
        reset_h()
        return f"Error: {str(e)}", history, "", []

# Function to handle text input
def respond(message, model_selector):
    global history
    try:
        if not message.strip():
            return "", history
        
        # Prepare messages including the current user prompt
        messages = history.copy()
        messages.append({"role": "user", "content": message})

        # Check if the selected model is a Groq model
        groq_models = ["deepseek-r1-distill-llama-70b", "llama3-70b-8192", "llama-3.3-70b-specdec"]
        if model_selector in groq_models:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model_selector,
            )   
            response = chat_completion.choices[0].message.content
        else:
            chat = ollama.chat(model=model_selector, messages=messages)
            response = chat['message']['content']
        if "deepseek" in model_selector.lower():
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        # Update history with user's question and the response
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        print(response)
        return "", history
    except Exception as e:
        reset_h()
        return f"Error: {str(e)}", history

# Set user and bot icons

current_dir = os.path.dirname(__file__)
user_icon = f"{current_dir}/pictures/ghibli_capibara.jpeg"
bot_icon = f"{current_dir}/pictures/japanese-lama-ghibli-artstyle.jpeg"

# Set markdown text
markitext = """ 
# Welcome to the Audio & PDF RAG Chatbot!

This chatbot enables you to ask questions about the contents of PDF documents or audio files. 
You can use either the text input or upload a file to get started.
"""

with gr.Blocks() as demo:
    marki = gr.Markdown(markitext)
    chatbot = gr.Chatbot(avatar_images=(user_icon, bot_icon), 
                        elem_id="chatbot",
                        group_consecutive_messages=True,
                        resizable=True,
                        show_copy_button=True,
                        type="messages",
                        show_share_button=True,
                        layout="panel")

    model_selector = gr.Dropdown(
        choices=["deepseek-r1-distill-llama-70b", "llama3-70b-8192", "llama-3.3-70b-specdec", "deepseek-r1:14b", "llama3:8b"],
        label="Select Model"
    )
    msg = gr.Textbox(label="Question without RAG")
    chunk_size_slider = gr.Slider(minimum=1000, maximum=5000, step=200, label="Chunk Size", interactive=True)
    overlap_slider = gr.Slider(minimum=0, maximum=200, step=10, label="Chunk Overlap", interactive=True)
    rag = gr.Textbox(label="Type Question for Rag",
                 interactive=True,
                 info="Type your question here before uploading a file")
    file = gr.File()
    
    file.upload(
        fn=upload_file,
        inputs=[file, rag, chunk_size_slider, overlap_slider, model_selector],
        outputs=[msg, chatbot, rag, file],
    )
    
    msg.submit(
        fn=respond,
        inputs=[msg, model_selector],
        outputs=[msg, chatbot]
    )
    
    clear = gr.ClearButton([msg, chatbot])
    clear_history = gr.Button("Clear History")
    clear_history.click(fn=reset_h)

demo.launch()
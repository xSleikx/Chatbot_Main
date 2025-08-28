# importing libraries 
import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama

AudioSegment.converter = "C:\\ffmpeg\\ffmpeg.exe"
AudioSegment.ffmpeg = "C:\\ffmpeg\\ffmpeg.exe"
AudioSegment.ffprobe ="C:\\ffmpeg\\ffprobe.exe"

# create a speech recognition object
r = sr.Recognizer()

# a function to recognize speech in the audio file, ja-JP for now
def transcribe_audio(path):
    # use the audio file as the audio source
    with sr.AudioFile(path) as source:
        audio_listened = r.record(source)
        # try converting it to text in Japanese (Japan) ja-JP
        # text = r.recognize_google(audio_listened)
        text = r.recognize_google(audio_listened, language="ja-JP")
    return text

# a function that splits the audio file into chunks on silence
# and applies speech recognition
def get_large_audio_transcription_on_silence(path):
    """Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks"""
    # open the audio file using pydub
    sound = AudioSegment.from_file(path)  
    # split audio sound where silence is 500 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 500,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        try:
            text = transcribe_audio(chunk_filename)
        except sr.UnknownValueError as e:
            print("Error:", str(e))
        else:
            text = f"{text.capitalize()}. "
            print(chunk_filename, ":", text)
            whole_text += text
    # return the text for all chunks detected
    print(whole_text)
    return whole_text


def load_and_retrieve_audio_docs(path, question, chunk_size_slider, overlap_slider):
    text = get_large_audio_transcription_on_silence(path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_slider, chunk_overlap=overlap_slider)
    docs = text_splitter.split_text(text)
    # Create Ollama embeddings and vector store
    embeddings = OllamaEmbeddings(model="llama3")
    print(embeddings)
    vectorstore = FAISS.from_texts(docs, embeddings)

    # Retrieve relevant documents based on the question
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(question)
    return retrieved_docs
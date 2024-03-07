import streamlit as st
import os
from pytube import YouTube
from openai import OpenAI
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from io import BytesIO
from pydub import AudioSegment
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import re
import assemblyai as aai
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from htmlTemplates import css, bot_template, user_template

# Load environment variables
load_dotenv()

# Set API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
aai_api_key = os.getenv("ASSEMBLYAI_API_KEY")

def extract_video_id(video_url):
    try:
        # Extract video ID from the YouTube URL
        video_id = YouTube(video_url).video_id
        return video_id
    except Exception as e:
        st.error(f"Error extracting video ID: {str(e)}")
        return None

# Download YouTube video
def download_youtube_video(video_url, selected_quality):
    try:
        yt = YouTube(video_url)
        available_streams = yt.streams.filter(res=selected_quality)

        if not available_streams:
            st.error(f"No streams available for the selected quality ({selected_quality}). Try another quality.")
            return None

        best_stream = available_streams.first()  # Pick the first available stream
        download_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        file_path = best_stream.download(download_dir)

        st.success(f"Video downloaded successfully at: {file_path}")
        return file_path
    except Exception as e:
        st.error(f"Error downloading video: {str(e)}")
        return None

# Download YouTube audio
def download_youtube_audio(video_url):
    try:
        yt = YouTube(video_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        download_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        sanitized_title = sanitize_filename(yt.title)
        audio_file_path = os.path.join(download_dir, f"{sanitized_title}.mp3")
        audio_stream.download(download_dir, filename=f"{sanitized_title}.mp3")

        st.success(f"Audio downloaded successfully at: {audio_file_path}")
        return audio_file_path
    except Exception as e:
        st.error(f"Error downloading audio: {str(e)}")
        return None

# Sanitize filename
def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '', filename)

# Get text as PDF
def get_text_as_pdf(video_url):
    try:
        yt = YouTube(video_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        download_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        sanitized_title = sanitize_filename(yt.title)
        temp_audio_path = os.path.join(download_dir, f"{sanitized_title}.mp3")
        audio_stream.download(download_dir, filename=f"{sanitized_title}.mp3")

        aai.settings.api_key = aai_api_key
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(temp_audio_path)

        if transcript.status == aai.TranscriptStatus.error:
            st.error(transcript.error)
        else:
            pdf_content = BytesIO()
            c = canvas.Canvas(pdf_content)
            c.drawString(72, 800, transcript.text)
            c.save()
            return pdf_content

        os.remove(temp_audio_path)

    except Exception as e:
        st.error(f"Error getting text: {str(e)}")

# Get text from PDF
def get_pdf_text(pdf_file):
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

# Get chunks from text
def get_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=100,
        chunk_overlap=50,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks

# Get vector from chunks
def get_vector(chunks, index_name="myind"):
    embedding = OpenAIEmbeddings(api_key=openai_api_key)
    vector_store = PineconeVectorStore.from_texts(chunks, embedding, index_name=index_name)
    return vector_store

# Create conversation chain
def get_conservation_chain(vectorstore):
    model = ChatOpenAI(api_key=openai_api_key, temperature=0.2)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=model, memory=memory, retriever=vectorstore.as_retriever()
    )
    return conversation_chain

# Handle user question
def handle_user_question(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True) if i % 2 == 0 else st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
def set_background_color(color):
    # Create custom CSS to set the background color
    custom_css = f"""
        <style>
            body {{
                background-color: {color};
            }}
        </style>
    """

    # Apply the custom CSS using st.markdown
    st.markdown(custom_css, unsafe_allow_html=True)
def main():
    load_dotenv()
    st.set_page_config(
        page_title="YouTube Video Chat and Downloader",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    set_background_color("black")
    st.title("YouTube Video Chat and Downloader")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    video_url = st.text_input("Paste YouTube video URL:")
    if video_url:
        video_id = extract_video_id(video_url)
        if video_id:
            st.video(f"https://www.youtube.com/embed/{video_id}")

    st.sidebar.title("Select any option to proceed:")

    video_downloaded = False
    if st.sidebar.button("Download Video"):
        if not video_url:
            st.warning("Please enter a YouTube URL to download the video.")
        else:
            selected_quality = st.selectbox("Select Video Quality:", ["720p", "1080p", "480p", "360p", "240p", "144p"])
            if selected_quality:
                with st.spinner("Downloading Video..."):
                    downloaded_file_path = download_youtube_video(video_url, selected_quality)
                    if downloaded_file_path:
                        video_downloaded = True

    if video_downloaded:
        # Provide a download button for the user
        file_content = open(downloaded_file_path, 'rb').read()
        st.download_button(
            label="Click here to download",
            data=file_content,
            file_name=os.path.basename(downloaded_file_path),
            key="download_button",
        )
    audio_downloaded = False
    if st.sidebar.button("Download Audio"):
        if not video_url:
            st.warning("Please enter a YouTube URL to download the Audio.")
        else:
            with st.spinner("Downloading Audio..."):
                downloaded_audio_path = download_youtube_audio(video_url)
                if downloaded_audio_path:
                    audio_downloaded = True
    if audio_downloaded:
        # Provide a download button for the user
        file_content = open(downloaded_audio_path, 'rb').read()
        st.download_button(
            label="Click here to download",
            data=file_content,
            file_name=os.path.basename(downloaded_audio_path),
            key="download_button",
        )
    if st.sidebar.button("Chat with Content"):
        if not video_url:
            st.warning("Please enter a YouTube URL to chat with the video.")
        else:
            with st.spinner("Processing"):
                pdf_file = get_text_as_pdf(video_url)
                if pdf_file is not None and pdf_file.getvalue():
                    st.success("Transcription successful.")
                    raw_text = get_pdf_text(pdf_file)
                    chunks = get_chunks(raw_text)
                    vector_store = get_vector(chunks)
                    st.session_state.conversation = get_conservation_chain(vector_store)

    user_question = st.text_input("Ask a question")
    if user_question:
        if st.session_state.conversation is not None:
            handle_user_question(user_question)
        else:
            st.warning("Please upload a video URL and click on chat with video content to start the conversation")

    st.write(user_template.replace("{{MSG}}", " User"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", " Chatbot"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()

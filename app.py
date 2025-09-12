import streamlit as st
from openai import OpenAI
import re
import tempfile
import os
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from pytube import YouTube

# ------------------------------------------------
# STREAMLIT PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="📺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------
def extract_video_id(url: str):
    """Extract YouTube video ID from URL"""
    patterns = [
        r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/)|youtu\.be\/)([^&?\/\s]{11})",
        r"^([^&?\/\s]{11})$"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_transcript_youtube(video_id: str):
    """Try fetching transcript via youtube-transcript-api"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join([x["text"] for x in transcript_list if x["text"].strip()])
        return transcript, "Success (captions)"
    except TranscriptsDisabled:
        return None, "Transcripts are disabled for this video."
    except NoTranscriptFound:
        return None, "No transcript found for this video."
    except Exception as e:
        return None, f"Error fetching transcript: {str(e)}"


def get_transcript_whisper(video_id: str, client: OpenAI):
    """Fallback: download audio and transcribe with Whisper API"""
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        audio_stream = yt.streams.filter(only_audio=True).first()

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        audio_stream.download(filename=tmp_file.name)

        with open(tmp_file.name, "rb") as f:
            transcript_response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )

        os.unlink(tmp_file.name)  # cleanup
        return transcript_response.text, "Success (Whisper)"
    except Exception as e:
        return None, f"Whisper transcription failed: {str(e)}"


def get_transcript(video_id: str, client: OpenAI):
    """Try captions first, fallback to Whisper if needed"""
    transcript, status = get_transcript_youtube(video_id)
    if transcript:
        return transcript, status

    transcript, status = get_transcript_whisper(video_id, client)
    return transcript, status


def summarize_transcript(transcript: str, client: OpenAI, model="gpt-3.5-turbo"):
    """Summarize transcript using GPT"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes YouTube transcripts."},
                {"role": "user", "content": f"Summarize this transcript clearly and concisely:\n\n{transcript}"}
            ],
            max_tokens=600,
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error during summarization: {str(e)}"


# ------------------------------------------------
# MAIN APP
# ------------------------------------------------
def main():
    st.title("📺 YouTube Video Summarizer")
    st.markdown("Paste a YouTube link and get an AI-powered summary.")

    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)

    url = st.text_input("YouTube Video URL", placeholder="Paste YouTube URL here...")
    generate_btn = st.button("Generate Summary")

    if generate_btn and url:
        with st.spinner("Extracting video ID..."):
            video_id = extract_video_id(url)

        if not video_id:
            st.error("Invalid YouTube URL. Please check and try again.")
            return

        st.video(f"https://www.youtube.com/watch?v={video_id}")

        with st.spinner("Fetching transcript..."):
            transcript, status = get_transcript(video_id, client)

        if not transcript:
            st.error(f"Could not retrieve transcript: {status}")
            return

        st.success(f"Transcript retrieved! ({status})")

        with st.expander("View Transcript"):
            st.text_area("Transcript", transcript, height=300)

        with st.spinner("Generating summary..."):
            summary = summarize_transcript(transcript, client)

        st.markdown("### 📝 Summary")
        st.write(summary)

        if st.button("Copy Summary"):
            st.code(summary)
            st.success("Summary copied to clipboard!")


if __name__ == "__main__":
    main()

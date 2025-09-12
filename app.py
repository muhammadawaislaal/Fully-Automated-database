import streamlit as st
import openai
import re
import requests
import tempfile
import os
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import tiktoken

# ----------------- CONFIG -----------------
st.set_page_config(page_title="YouTube Video Summarizer", page_icon="📺", layout="centered")

if "summary" not in st.session_state:
    st.session_state.summary = None

# ----------------- HELPERS -----------------
def extract_video_id(url: str):
    """Extract YouTube video ID"""
    patterns = [
        r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/)|youtu\.be\/)([^&?\/\s]{11})",
        r"^([^&?\/\s]{11})$"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript_from_youtube(video_id: str):
    """Try fetching transcript using YouTubeTranscriptApi"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join([x["text"] for x in transcript_list if x["text"].strip()])
        return transcript, "Success"
    except TranscriptsDisabled:
        return None, "Transcripts are disabled for this video."
    except NoTranscriptFound:
        return None, "No transcript found for this video."
    except Exception as e:
        return None, f"Error fetching transcript: {str(e)}"

def get_transcript_with_whisper(video_url: str, api_key: str):
    """Download audio and transcribe with Whisper"""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.mp3")

            # Download audio
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": audio_path,
                "quiet": True,
                "noplaylist": True,
                "extractaudio": True,
                "audioformat": "mp3"
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])

            # OpenAI Whisper transcription
            openai.api_key = api_key
            with open(audio_path, "rb") as f:
                transcript = openai.Audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
            return transcript.text, "Transcribed with Whisper"
    except Exception as e:
        return None, f"Whisper failed: {str(e)}"

def count_tokens(text):
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    except Exception:
        return len(text.split())

def split_text(text, max_tokens=3000):
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = encoding.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i+max_tokens]
            chunks.append(encoding.decode(chunk_tokens))
        return chunks
    except Exception:
        words = text.split()
        chunks, current = [], []
        count = 0
        for w in words:
            if count + len(w.split()) > max_tokens:
                chunks.append(" ".join(current))
                current = [w]
                count = len(w.split())
            else:
                current.append(w)
                count += len(w.split())
        if current:
            chunks.append(" ".join(current))
        return chunks

def summarize_transcript(transcript, api_key, model="gpt-3.5-turbo"):
    openai.api_key = api_key
    token_count = count_tokens(transcript)

    if token_count > 3000:
        chunks = split_text(transcript)
        summaries = []
        for i, chunk in enumerate(chunks):
            with st.spinner(f"Summarizing part {i+1}/{len(chunks)}..."):
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes YouTube transcripts."},
                        {"role": "user", "content": f"Summarize this transcript section:\n\n{chunk}"}
                    ],
                    max_tokens=500,
                    temperature=0.3
                )
                summaries.append(resp.choices[0].message.content.strip())
        final_prompt = "Combine these summaries into one coherent video summary:\n\n" + "\n\n".join(summaries)
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You write comprehensive video summaries."},
                {"role": "user", "content": final_prompt}
            ],
            max_tokens=600,
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()
    else:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes YouTube transcripts."},
                {"role": "user", "content": transcript}
            ],
            max_tokens=800,
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()

# ----------------- UI -----------------
def main():
    st.title("📺 YouTube Video Summarizer")
    st.markdown("Summarize **any YouTube video** with captions or Whisper fallback")

    api_key = st.text_input("🔑 OpenAI API Key", type="password")
    url = st.text_input("🎥 YouTube URL", placeholder="Paste a YouTube link...")
    if st.button("Generate Summary", disabled=not api_key):
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Invalid YouTube URL")
            return

        st.video(f"https://www.youtube.com/watch?v={video_id}")

        with st.spinner("Fetching transcript..."):
            transcript, status = get_transcript_from_youtube(video_id)

        if not transcript:
            st.warning(f"{status} → Falling back to Whisper...")
            with st.spinner("Downloading audio & transcribing with Whisper..."):
                transcript, status = get_transcript_with_whisper(url, api_key)

        if not transcript:
            st.error(f"Could not retrieve transcript: {status}")
            return

        st.success("Transcript ready ✅")
        with st.expander("View transcript"):
            st.text(transcript[:2000] + "..." if len(transcript) > 2000 else transcript)

        with st.spinner("Generating summary..."):
            summary = summarize_transcript(transcript, api_key)

        if summary:
            st.session_state.summary = summary
            st.markdown("### 📝 Summary")
            st.write(summary)

if __name__ == "__main__":
    main()

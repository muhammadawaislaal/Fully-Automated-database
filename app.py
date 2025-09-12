import streamlit as st
from openai import OpenAI
import re
import tempfile
import os
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from pytube import YouTube
from pydub import AudioSegment

st.set_page_config(page_title="YouTube Video Summarizer", page_icon="📺")

def extract_video_id(url: str):
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
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join([x["text"] for x in transcript_list if x["text"].strip()])
        return transcript, "Success (captions)"
    except (TranscriptsDisabled, NoTranscriptFound):
        return None, "No captions available"
    except Exception as e:
        return None, f"Error: {e}"

def get_transcript_whisper(video_id: str, client: OpenAI):
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        audio_stream = yt.streams.filter(only_audio=True).first()

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        audio_stream.download(filename=tmp_file.name)

        audio = AudioSegment.from_file(tmp_file.name)
        os.unlink(tmp_file.name)

        # lower bitrate → smaller file
        chunk_length_ms = 5 * 60 * 1000
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

        transcripts = []
        for i, chunk in enumerate(chunks):
            chunk_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            chunk.export(chunk_file.name, format="mp3", bitrate="64k")

            with open(chunk_file.name, "rb") as f:
                resp = client.audio.transcriptions.create(model="whisper-1", file=f)
                transcripts.append(resp.text)

            os.unlink(chunk_file.name)

        return " ".join(transcripts), "Success (Whisper)"
    except Exception as e:
        return None, f"Whisper failed: {e}"

def get_transcript(video_id: str, client: OpenAI):
    transcript, status = get_transcript_youtube(video_id)
    if transcript:
        return transcript, status
    return get_transcript_whisper(video_id, client)

def summarize_transcript(transcript: str, client: OpenAI, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes YouTube transcripts."},
            {"role": "user", "content": f"Summarize this transcript:\n\n{transcript}"}
        ],
        max_tokens=600,
        temperature=0.4
    )
    return response.choices[0].message.content.strip()

def main():
    st.title("📺 YouTube Video Summarizer")

    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)

    url = st.text_input("YouTube URL")
    if st.button("Generate Summary") and url:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Invalid URL")
            return

        st.video(f"https://www.youtube.com/watch?v={video_id}")

        transcript, status = get_transcript(video_id, client)
        if not transcript:
            st.error(f"Could not retrieve transcript: {status}")
            return

        st.success(status)
        with st.expander("Transcript"):
            st.text_area("Transcript", transcript, height=300)

        summary = summarize_transcript(transcript, client)
        st.subheader("📝 Summary")
        st.write(summary)

if __name__ == "__main__":
    main()

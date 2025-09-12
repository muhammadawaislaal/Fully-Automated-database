import streamlit as st
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import re
import tiktoken

# -----------------------
# Streamlit Page Config
# -----------------------
st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="📺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------
# Session State
# -----------------------
if "summary" not in st.session_state:
    st.session_state.summary = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "api_key_configured" not in st.session_state:
    st.session_state.api_key_configured = False

# -----------------------
# Helper Functions
# -----------------------
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


def get_transcript(video_id: str):
    """Get transcript, fallback to auto-generated or non-English"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try manual English transcript
        try:
            t = transcript_list.find_transcript(['en'])
            transcript = t.fetch()
            text = " ".join([x["text"] for x in transcript if x["text"].strip()])
            return text, "en", "Success"
        except:
            pass

        # Try auto-generated English
        try:
            t = transcript_list.find_generated_transcript(['en'])
            transcript = t.fetch()
            text = " ".join([x["text"] for x in transcript if x["text"].strip()])
            return text, "en", "Success"
        except:
            pass

        # Try ANY transcript in another language
        for t in transcript_list:
            try:
                transcript = t.fetch()
                text = " ".join([x["text"] for x in transcript if x["text"].strip()])
                return text, t.language_code, "Non-English"
            except:
                continue

        return None, None, "No transcript available"

    except TranscriptsDisabled:
        return None, None, "Transcripts are disabled for this video."
    except NoTranscriptFound:
        return None, None, "No transcript found for this video."
    except Exception as e:
        return None, None, f"Error fetching transcript: {str(e)}"


def count_tokens(text: str):
    """Count tokens with tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    except:
        return len(text.split())


def split_text(text, max_tokens=3000):
    """Split text into chunks"""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(encoding.decode(chunk_tokens))
    return chunks


def translate_text(text, api_key, source_lang):
    """Translate non-English transcript into English"""
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are a translator. Translate from {source_lang} to English."},
            {"role": "user", "content": text}
        ],
        max_tokens=800,
        temperature=0.3
    )
    return resp.choices[0].message.content.strip()


def summarize_transcript(transcript, api_key, model="gpt-3.5-turbo"):
    """Summarize transcript using OpenAI"""
    client = OpenAI(api_key=api_key)
    token_count = count_tokens(transcript)

    if token_count > 3000:
        chunks = split_text(transcript)
        summaries = []
        for i, chunk in enumerate(chunks):
            with st.spinner(f"Summarizing part {i + 1}/{len(chunks)}..."):
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes YouTube transcripts."},
                        {"role": "user", "content": f"Summarize this part:\n\n{chunk}"}
                    ],
                    max_tokens=500,
                    temperature=0.3
                )
                summaries.append(resp.choices[0].message.content.strip())

        combined_text = "\n\n".join(summaries)
        final_resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You combine partial summaries into a full summary."},
                {"role": "user", "content": f"Combine these summaries into a coherent overall summary:\n\n{combined_text}"}
            ],
            max_tokens=600,
            temperature=0.3
        )
        return final_resp.choices[0].message.content.strip()
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes YouTube transcripts."},
                {"role": "user", "content": f"Please summarize this video transcript:\n\n{transcript}"}
            ],
            max_tokens=800,
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()


# -----------------------
# Main App
# -----------------------
def main():
    st.title("📺 YouTube Video Summarizer")
    st.markdown("Generate AI-powered summaries of YouTube videos using their transcripts")

    # API Key
    if "OPENAI_API_KEY" in st.secrets:
        st.session_state.api_key_configured = True
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
        if api_key:
            st.session_state.api_key_configured = True

    # Input
    url = st.text_input("YouTube Video URL", placeholder="Paste YouTube URL here...")
    generate_btn = st.button("Generate Summary", disabled=not st.session_state.api_key_configured)

    if generate_btn and url:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Invalid YouTube URL.")
            return

        st.video(f"https://www.youtube.com/watch?v={video_id}")
        transcript, lang, msg = get_transcript(video_id)

        if not transcript:
            st.error(f"Could not retrieve transcript: {msg}")
            st.info("⚠️ Try a different video. Not all YouTube videos have transcripts.")
            return

        # Translate if needed
        if lang != "en":
            st.warning(f"Transcript is in {lang}. Translating to English...")
            transcript = translate_text(transcript[:3000], api_key, lang)

        st.success("Transcript ready!")
        with st.expander("View Transcript"):
            st.text(transcript[:1000] + "..." if len(transcript) > 1000 else transcript)

        summary = summarize_transcript(transcript, api_key)
        if summary:
            st.session_state.summary = summary
            st.markdown("### 📝 Summary")
            st.write(summary)

    if st.session_state.summary:
        if st.button("Copy Summary"):
            st.code(st.session_state.summary)


if __name__ == "__main__":
    main()

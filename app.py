import streamlit as st
import openai
from youtube_transcript_api import YouTubeTranscriptApi
import re
import tiktoken

# Set page configuration
st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="📺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'video_id' not in st.session_state:
    st.session_state.video_id = None
if 'video_title' not in st.session_state:
    st.session_state.video_title = None

def extract_video_id(url):
    """Extract YouTube video ID from URL"""
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/)|youtu\.be\/)([^&?\/\s]{11})',
        r'^([^&?\/\s]{11})$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id):
    """Fetch transcript for a YouTube video"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([entry['text'] for entry in transcript_list])
        return transcript
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None

def count_tokens(text):
    """Count tokens in text using tiktoken for GPT-3.5"""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def split_text(text, max_tokens=3000):
    """Split text into chunks that don't exceed the token limit"""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunks.append(encoding.decode(chunk_tokens))
    
    return chunks

def summarize_transcript(transcript, api_key, model="gpt-3.5-turbo"):
    """Summarize transcript using OpenAI API"""
    # Check token count and split if necessary
    token_count = count_tokens(transcript)
    
    if token_count > 3000:
        chunks = split_text(transcript)
        summaries = []
        
        for i, chunk in enumerate(chunks):
            with st.spinner(f"Summarizing part {i+1}/{len(chunks)}..."):
                prompt = f"Please provide a detailed summary of the following video transcript section:\n\n{chunk}"
                
                try:
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that summarizes YouTube video transcripts."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500,
                        temperature=0.3
                    )
                    summaries.append(response.choices[0].message.content.strip())
                except Exception as e:
                    st.error(f"Error during API call: {str(e)}")
                    return None
        
        # Combine and summarize the summaries
        combined_summaries = "\n\n".join(summaries)
        final_prompt = f"Please combine these section summaries into a coherent overall summary of the video:\n\n{combined_summaries}"
        
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes YouTube video transcripts."},
                    {"role": "user", "content": final_prompt}
                ],
                max_tokens=600,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Error during final API call: {str(e)}")
            return None
    else:
        prompt = f"Please provide a comprehensive summary of the following YouTube video transcript. Include main points, key insights, and conclusions:\n\n{transcript}"
        
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes YouTube video transcripts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Error during API call: {str(e)}")
            return None

def main():
    st.title("📺 YouTube Video Summarizer")
    st.markdown("Generate AI-powered summaries of YouTube videos using their URLs")
    
    # Sidebar for API key input
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", 
                               help="Enter your OpenAI API key. This is required to generate summaries.")
        st.markdown("---")
        st.markdown("### How to use")
        st.markdown("""
        1. Enter your OpenAI API key
        2. Paste a YouTube URL
        3. Click 'Generate Summary'
        4. View and copy your summary
        """)
        st.markdown("---")
        st.markdown("### Note")
        st.markdown("This tool extracts the transcript from YouTube videos and uses AI to generate a summary. Not all videos have available transcripts.")
    
    # Main content area
    url = st.text_input("YouTube Video URL", placeholder="Paste YouTube URL here...")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        generate_btn = st.button("Generate Summary", disabled=not api_key)
    with col2:
        if st.session_state.summary:
            copy_btn = st.button("Copy Summary")
    
    if generate_btn and url:
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
            return
            
        openai.api_key = api_key
        
        with st.spinner("Extracting video ID..."):
            video_id = extract_video_id(url)
            
        if not video_id:
            st.error("Invalid YouTube URL. Please check the URL and try again.")
            return
            
        st.session_state.video_id = video_id
        embed_url = f"https://www.youtube.com/embed/{video_id}"
        
        st.markdown("---")
        st.markdown(f"### Video Preview")
        st.video(url)
        
        with st.spinner("Fetching transcript..."):
            transcript = get_transcript(video_id)
            
        if not transcript:
            st.error("Could not retrieve transcript for this video. It may not have captions available.")
            return
            
        with st.expander("View Raw Transcript"):
            st.text(transcript[:1000] + "..." if len(transcript) > 1000 else transcript)
            
        with st.spinner("Generating summary..."):
            summary = summarize_transcript(transcript, api_key)
            
        if summary:
            st.session_state.summary = summary
            st.markdown("---")
            st.markdown("### 📝 Summary")
            st.write(summary)
            
    elif generate_btn and not url:
        st.error("Please enter a YouTube URL.")
        
    if copy_btn and st.session_state.summary:
        st.code(st.session_state.summary)
        st.success("Summary copied to clipboard!")

if __name__ == "__main__":
    main()

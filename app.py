import streamlit as st
import openai
import re
import tiktoken
import requests
import json

# Try to import YouTubeTranscriptApi with compatibility handling
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, VideoUnavailable, TooManyRequests
    
    # Handle different versions of the library
    try:
        from youtube_transcript_api._errors import NoTranscriptFound
        NO_TRANSCRIPT_ERROR = NoTranscriptFound
    except ImportError:
        try:
            from youtube_transcript_api._errors import NoTranscriptAvailable
            NO_TRANSCRIPT_ERROR = NoTranscriptAvailable
        except ImportError:
            # Fallback if neither exists
            class NoTranscriptFound(Exception):
                pass
            NO_TRANSCRIPT_ERROR = NoTranscriptFound
            
except ImportError:
    st.error("youtube-transcript-api is not installed. Please add it to your requirements.txt")
    # Create dummy classes to prevent further errors
    class YouTubeTranscriptApi:
        @staticmethod
        def list_transcripts(video_id):
            raise Exception("youtube-transcript-api not installed")
    
    class TranscriptsDisabled(Exception):
        pass
        
    class VideoUnavailable(Exception):
        pass
        
    class TooManyRequests(Exception):
        pass
        
    class NO_TRANSCRIPT_ERROR(Exception):
        pass

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
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = False

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
    """Get transcript using YouTube Transcript API"""
    try:
        # Try to get English transcript first
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get English transcript
        try:
            transcript = transcript_list.find_transcript(['en'])
            translated_transcript = transcript.translate('en')
            transcript_data = translated_transcript.fetch()
        except:
            # If English not available, get the first available transcript
            transcript = transcript_list.find_transcript([t.language_code for t in transcript_list])
            translated_transcript = transcript.translate('en')
            transcript_data = translated_transcript.fetch()
        
        # Combine all text elements
        text = ' '.join([item['text'] for item in transcript_data])
        return text, "Success"
    
    except TranscriptsDisabled:
        return None, "Transcripts are disabled for this video"
    except NO_TRANSCRIPT_ERROR:
        return None, "No transcript available for this video"
    except VideoUnavailable:
        return None, "Video is unavailable"
    except TooManyRequests:
        return None, "Too many requests. Please try again later."
    except Exception as e:
        return None, f"Error: {str(e)}"

def count_tokens(text):
    """Count tokens in text using tiktoken for GPT-3.5"""
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    except:
        # Fallback: approximate token count
        return len(text.split())

def split_text(text, max_tokens=3000):
    """Split text into chunks that don't exceed the token limit"""
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = encoding.encode(text)
        
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i+max_tokens]
            chunks.append(encoding.decode(chunk_tokens))
        
        return chunks
    except:
        # Fallback: split by words
        words = text.split()
        chunks = []
        current_chunk = []
        current_count = 0
        
        for word in words:
            if current_count + len(word.split()) > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_count = len(word.split())
            else:
                current_chunk.append(word)
                current_count += len(word.split())
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

def summarize_transcript(transcript, api_key, model="gpt-3.5-turbo"):
    """Summarize transcript using OpenAI API"""
    # Set the API key
    openai.api_key = api_key
    
    # Check token count and split if necessary
    token_count = count_tokens(transcript)
    
    prompt = f"""Please provide a comprehensive summary of the following YouTube video transcript. 
    Include main points, key insights, and conclusions. Structure your response clearly:
    
    {transcript}"""
    
    if token_count > 3000:
        chunks = split_text(transcript)
        summaries = []
        
        for i, chunk in enumerate(chunks):
            with st.spinner(f"Summarizing part {i+1}/{len(chunks)}..."):
                chunk_prompt = f"Please summarize this section of a video transcript:\n\n{chunk}"
                
                try:
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that summarizes YouTube video transcripts."},
                            {"role": "user", "content": chunk_prompt}
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
                    {"role": "system", "content": "You create comprehensive video summaries."},
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
    
    # Check for API key in secrets
    if 'OPENAI_API_KEY' in st.secrets:
        st.session_state.api_key_configured = True
        api_key = st.secrets['OPENAI_API_KEY']
    else:
        st.session_state.api_key_configured = False
        api_key = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        if not st.session_state.api_key_configured:
            api_key = st.text_input("OpenAI API Key", type="password", 
                                   help="Enter your OpenAI API key. This is required to generate summaries.")
            if api_key:
                st.session_state.api_key_configured = True
        else:
            st.success("API key loaded from secrets")
            if st.button("Use different API key"):
                st.session_state.api_key_configured = False
                st.experimental_rerun()
        
        st.markdown("---")
        st.markdown("### How to use")
        st.markdown("""
        1. Enter your OpenAI API key (if not in secrets)
        2. Paste a YouTube URL
        3. Click 'Generate Summary'
        4. View and copy your summary
        """)
        st.markdown("---")
        st.markdown("### Note")
        st.markdown("""
        - This tool extracts the transcript from YouTube videos
        - Uses AI to generate a summary
        - Not all videos have available transcripts
        - Try videos that definitely have captions/subtitles
        - Longer videos may take more time to process
        """)
        
        st.markdown("### Test Videos with Captions")
        st.markdown("Try these videos with known transcripts:")
        st.markdown("- https://www.youtube.com/watch?v=Lp7E973zozs (TED Talk)")
        st.markdown("- https://www.youtube.com/watch?v=JcP7wX08vq0 (Google I/O)")
        st.markdown("- https://www.youtube.com/watch?v=8S0FDjFBj8o (Microsoft Build)")
    
    # Main content area
    url = st.text_input("YouTube Video URL", placeholder="Paste YouTube URL here...")
    
    # Example video buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("TED Talk Example"):
            st.session_state.url = "https://www.youtube.com/watch?v=Lp7E973zozs"
            st.experimental_rerun()
    with col2:
        if st.button("Google I/O Example"):
            st.session_state.url = "https://www.youtube.com/watch?v=JcP7wX08vq0"
            st.experimental_rerun()
    with col3:
        if st.button("Microsoft Build Example"):
            st.session_state.url = "https://www.youtube.com/watch?v=8S0FDjFBj8o"
            st.experimental_rerun()
    
    # Get URL from session state if set by buttons
    if hasattr(st.session_state, 'url'):
        url = st.session_state.url
    
    generate_btn = st.button("Generate Summary", disabled=not st.session_state.api_key_configured)
    
    if generate_btn and url:
        if not st.session_state.api_key_configured:
            st.error("Please enter your OpenAI API key in the sidebar.")
            return
            
        with st.spinner("Extracting video ID..."):
            video_id = extract_video_id(url)
            
        if not video_id:
            st.error("Invalid YouTube URL. Please check the URL and try again.")
            return
            
        st.session_state.video_id = video_id
        
        st.markdown("---")
        st.markdown(f"### Video Preview")
        st.video(f"https://www.youtube.com/watch?v={video_id}")
        
        with st.spinner("Fetching transcript..."):
            # Try to get transcript using YouTube Transcript API
            transcript, error_message = get_transcript(video_id)
            
        if not transcript:
            st.error(f"Could not retrieve transcript for this video. {error_message}")
            st.info("""
            **Tips for success:**
            - Try videos that definitely have captions (like the examples above)
            - Make sure the video has English captions available
            - Some videos have region-restricted transcripts
            - Try refreshing and trying again
            """)
            return
            
        st.success("Successfully retrieved transcript!")
            
        with st.expander("View Raw Transcript"):
            st.text(transcript[:1000] + "..." if len(transcript) > 1000 else transcript)
            
        with st.spinner("Generating summary (this may take a while for longer videos)..."):
            summary = summarize_transcript(transcript, api_key)
            
        if summary:
            st.session_state.summary = summary
            st.markdown("---")
            st.markdown("### 📝 Summary")
            st.write(summary)
            
    elif generate_btn and not url:
        st.error("Please enter a YouTube URL.")
    
    # Display copy button only if there's a summary
    if st.session_state.summary:
        if st.button("Copy Summary"):
            st.code(st.session_state.summary)
            st.success("Summary copied to clipboard!")

if __name__ == "__main__":
    main()

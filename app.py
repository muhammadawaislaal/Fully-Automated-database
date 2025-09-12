import streamlit as st
import openai
import re
import tiktoken
import requests
from urllib.parse import parse_qs, urlparse
import xml.etree.ElementTree as ET
import json

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

def get_captions_from_html(video_id):
    """Extract captions from YouTube HTML page"""
    try:
        # Fetch the YouTube page
        url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        html_content = response.text
        
        # Look for captions JSON data
        caption_patterns = [
            r'"captions":\s*({.*?}),"',
            r'"captionTracks":\s*(\[.*?\])',
            r'ytInitialPlayerResponse\s*=\s*({.*?});'
        ]
        
        for pattern in caption_patterns:
            match = re.search(pattern, html_content)
            if match:
                try:
                    captions_data = json.loads(match.group(1))
                    
                    # Try different JSON structures
                    if 'playerCaptionsTracklistRenderer' in captions_data:
                        caption_tracks = captions_data['playerCaptionsTracklistRenderer'].get('captionTracks', [])
                    elif 'captionTracks' in captions_data:
                        caption_tracks = captions_data['captionTracks']
                    elif isinstance(captions_data, list):
                        caption_tracks = captions_data
                    else:
                        continue
                    
                    # Find English captions
                    for track in caption_tracks:
                        if (track.get('languageCode') == 'en' and 
                            track.get('kind') != 'asr' and  # Prefer non-auto-generated
                            track.get('vssId') and 
                            '.en' in track.get('vssId', '')):
                            
                            caption_url = track.get('baseUrl')
                            if caption_url:
                                # Fetch the caption XML
                                caption_response = requests.get(caption_url)
                                if caption_response.status_code == 200:
                                    # Parse XML captions
                                    root = ET.fromstring(caption_response.content)
                                    transcript = ' '.join([elem.text for elem in root.findall('.//text') if elem.text])
                                    return transcript, "Success"
                    
                except json.JSONDecodeError:
                    continue
        
        return None, "No captions found in HTML"
    
    except Exception as e:
        return None, f"Error: {str(e)}"

def get_transcript_direct(video_id):
    """Direct method to get transcript using YouTube's internal API"""
    try:
        # Try to get transcript using YouTube's internal API endpoint
        transcript_url = f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en"
        
        response = requests.get(transcript_url)
        if response.status_code == 200:
            # Parse XML response
            root = ET.fromstring(response.content)
            transcript = ' '.join([elem.text for elem in root.findall('.//text') if elem.text])
            if transcript:
                return transcript, "Success"
        
        return None, "No transcript available via direct API"
    
    except Exception as e:
        return None, f"Error: {str(e)}"

def get_transcript_final(video_id):
    """Final attempt to get transcript using all available methods"""
    # Try direct API first
    transcript, error = get_transcript_direct(video_id)
    if transcript:
        return transcript, error
    
    # Try HTML extraction
    transcript, error = get_captions_from_html(video_id)
    if transcript:
        return transcript, error
    
    return None, "All methods failed. This video may not have captions available."

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
        st.markdown("- https://www.youtube.com/watch?v=Lp7E973zozs")
        st.markdown("- https://www.youtube.com/watch?v=JcP7wX08vq0")
        st.markdown("- https://www.youtube.com/watch?v=8S0FDjFBj8o")
    
    # Main content area
    url = st.text_input("YouTube Video URL", placeholder="Paste YouTube URL here...")
    
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
            # Try to get transcript using our final method
            transcript, error_message = get_transcript_final(video_id)
            
        if not transcript:
            st.error(f"Could not retrieve transcript for this video. {error_message}")
            st.info("""
            **Tips for success:**
            - Try videos that definitely have captions (like the examples in the sidebar)
            - Make sure the video has English captions available
            - Some videos have region-restricted transcripts
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

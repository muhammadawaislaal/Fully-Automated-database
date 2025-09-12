import streamlit as st
import openai
import re
import tiktoken
import requests
import json
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, parse_qs
import time

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
if 'transcript_method' not in st.session_state:
    st.session_state.transcript_method = None

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
    """Extract captions from YouTube page HTML - Improved version"""
    try:
        # Fetch YouTube page
        url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        html_content = response.text
        
        # Look for caption tracks in the HTML
        caption_pattern = r'"captionTracks":\s*(\[.*?\])'
        match = re.search(caption_pattern, html_content)
        
        if match:
            try:
                caption_data = json.loads(match.group(1))
                
                # Try to find English captions first
                for track in caption_data:
                    if (track.get('languageCode') == 'en' and 
                        track.get('vssId') and '.en' in track.get('vssId', '')):
                        
                        caption_url = track.get('baseUrl')
                        if caption_url:
                            # Add &fmt=json to get JSON format which is easier to parse
                            if '&fmt=' not in caption_url:
                                caption_url += '&fmt=json3'
                            
                            caption_response = requests.get(caption_url, headers=headers, timeout=10)
                            if caption_response.status_code == 200:
                                try:
                                    # Try to parse as JSON first
                                    caption_json = caption_response.json()
                                    events = caption_json.get('events', [])
                                    transcript_text = ""
                                    
                                    for event in events:
                                        if 'segs' in event:
                                            for seg in event['segs']:
                                                if 'utf8' in seg:
                                                    transcript_text += seg['utf8'] + " "
                                    
                                    if transcript_text.strip():
                                        return transcript_text, "Success (HTML/JSON method)"
                                except:
                                    # Fallback to XML parsing
                                    try:
                                        root = ET.fromstring(caption_response.content)
                                        text_elements = []
                                        
                                        for elem in root.iter('text'):
                                            if elem.text:
                                                text_elements.append(elem.text)
                                        
                                        if text_elements:
                                            transcript = " ".join(text_elements)
                                            return transcript, "Success (HTML/XML method)"
                                    except:
                                        continue
            except json.JSONDecodeError:
                pass
        
        return None, "No captions found in HTML"
    
    except Exception as e:
        return None, f"Error: {str(e)}"

def get_captions_via_youtube_transcript_api(video_id):
    """Try to get captions using youtube-transcript-api approach"""
    try:
        # This mimics what youtube-transcript-api does
        transcript_url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        response = requests.get(transcript_url, headers=headers, timeout=10)
        html_content = response.text
        
        # Try to find the captions configuration
        splitted_html = html_content.split('"captions":')
        
        if len(splitted_html) > 1:
            captions_json = json.loads(splitted_html[1].split(',"videoDetails"')[0].replace('\n', ''))
            caption_tracks = captions_json['playerCaptionsTracklistRenderer']['captionTracks']
            
            # Look for English captions
            for track in caption_tracks:
                if track.get('languageCode', '').startswith('en'):
                    caption_url = track.get('baseUrl', '')
                    if caption_url:
                        # Get the transcript
                        transcript_response = requests.get(caption_url, headers=headers, timeout=10)
                        if transcript_response.status_code == 200:
                            try:
                                # Parse XML
                                root = ET.fromstring(transcript_response.content)
                                text_elements = []
                                
                                for elem in root.iter('text'):
                                    if elem.text:
                                        text_elements.append(elem.text)
                                
                                if text_elements:
                                    transcript = " ".join(text_elements)
                                    return transcript, "Success (Transcript API method)"
                            except:
                                pass
        
        return None, "No captions found via Transcript API"
    
    except Exception as e:
        return None, f"Error: {str(e)}"

def get_transcript(video_id):
    """Main function to get transcript using multiple methods"""
    methods = [
        get_captions_via_youtube_transcript_api,
        get_captions_from_html
    ]
    
    for method in methods:
        transcript, error = method(video_id)
        if transcript:
            st.session_state.transcript_method = method.__name__
            return transcript, error
        
        # Add a small delay between attempts
        time.sleep(1)
    
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
        st.markdown("- https://www.youtube.com/watch?v=Lp7E973zozs (TED Talk)")
        st.markdown("- https://www.youtube.com/watch?v=JcP7wX08vq0 (Google I/O)")
        st.markdown("- https://www.youtube.com/watch?v=8S0FDjFBj8o (Microsoft Build)")
    
    # Main content area
    url = st.text_input("YouTube Video URL", placeholder="Paste YouTube URL here...")
    
    # Example video buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("TED Talk Example"):
            url = "https://www.youtube.com/watch?v=Lp7E973zozs"
    with col2:
        if st.button("Google I/O Example"):
            url = "https://www.youtube.com/watch?v=JcP7wX08vq0"
    with col3:
        if st.button("Microsoft Build Example"):
            url = "https://www.youtube.com/watch?v=8S0FDjFBj8o"
    
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
        
        with st.spinner("Fetching transcript (this may take a moment)..."):
            # Try to get transcript using our method
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
            
        st.success(f"Successfully retrieved transcript using {st.session_state.transcript_method}!")
            
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

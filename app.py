import streamlit as st
import openai
import re
import tiktoken
import requests
from bs4 import BeautifulSoup
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

def get_transcript_from_api(video_id):
    """
    Get transcript using a reliable API service
    This approach uses a third-party service to fetch YouTube transcripts
    """
    try:
        # Using a third-party API to get YouTube transcripts
        # Note: In production, you might want to use a paid service or YouTube's official API
        api_url = f"https://youtube-transcriptor.p.rapidapi.com/transcript"
        querystring = {"video_id": video_id, "lang": "en"}
        
        headers = {
            "X-RapidAPI-Key": "your-rapidapi-key-here",  # You would need to sign up for RapidAPI
            "X-RapidAPI-Host": "youtube-transcriptor.p.rapidapi.com"
        }
        
        response = requests.get(api_url, headers=headers, params=querystring)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('transcript'):
                return data['transcript'], "Success"
        
        return None, "Transcript not available via API"
    
    except:
        # Fallback to manual extraction
        return get_transcript_manual(video_id)

def get_transcript_manual(video_id):
    """
    Manual method to extract transcript from YouTube page
    This is a fallback method that works for many videos
    """
    try:
        # Fetch the YouTube page
        url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for transcript data in the page
        scripts = soup.find_all('script')
        transcript_data = None
        
        for script in scripts:
            if 'captionTracks' in str(script):
                # Extract JSON data from script tag
                script_text = str(script)
                start = script_text.find('{"captionTracks":')
                if start != -1:
                    end = script_text.find('};', start) + 1
                    if end != 0:
                        json_text = script_text[start:end]
                        try:
                            data = json.loads(json_text)
                            caption_tracks = data.get('captionTracks', [])
                            
                            # Find English captions
                            for track in caption_tracks:
                                if track.get('languageCode') == 'en' and track.get('kind') == 'asr':
                                    transcript_url = track.get('baseUrl')
                                    if transcript_url:
                                        # Fetch the transcript XML
                                        transcript_response = requests.get(transcript_url)
                                        if transcript_response.status_code == 200:
                                            transcript_soup = BeautifulSoup(transcript_response.text, 'xml')
                                            texts = transcript_soup.find_all('text')
                                            transcript = ' '.join([text.get_text() for text in texts])
                                            return transcript, "Success"
                        except:
                            continue
        
        return None, "Could not extract transcript from page"
    
    except Exception as e:
        return None, f"Error: {str(e)}"

def get_transcript_simple(video_id):
    """
    Simple method that works for many videos with captions
    """
    try:
        # Try to use youtube-transcript-api with correct method names
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            
            # Try different method names that might work
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                transcript = " ".join([entry['text'] for entry in transcript_list])
                return transcript, "Success"
            except:
                try:
                    # Some versions use list_transcripts
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                    transcript = transcript_list.find_transcript(['en']).fetch()
                    transcript_text = " ".join([entry['text'] for entry in transcript])
                    return transcript_text, "Success"
                except:
                    pass
        except:
            pass
        
        # If library methods fail, try manual extraction
        return get_transcript_manual(video_id)
    
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
        
        st.markdown("### Example Videos")
        st.markdown("Try these videos with known transcripts:")
        st.markdown("- https://www.youtube.com/watch?v=H14bBuluwB8")
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
            # Try to get transcript using our reliable method
            transcript, error_message = get_transcript_simple(video_id)
            
        if not transcript:
            st.error(f"Could not retrieve transcript for this video. {error_message}")
            st.info("""
            **Tips for success:**
            - Try videos that definitely have captions/subtitles
            - Try the example videos provided in the sidebar
            - Some videos have region-restricted transcripts
            - The video might not have captions available
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

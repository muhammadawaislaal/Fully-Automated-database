import streamlit as st
import openai
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import re
import tiktoken
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
if 'video_title' not in st.session_state:
    st.session_state.video_title = None
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

def get_transcript(video_id, languages=None):
    """Fetch transcript for a YouTube video with support for multiple languages"""
    try:
        if languages:
            # Try to get transcript in specified languages
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        else:
            # Try to get transcript in any available language
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        transcript = " ".join([entry['text'] for entry in transcript_list])
        return transcript, transcript_list[0]['language'] if transcript_list else "unknown"
    
    except TranscriptsDisabled:
        return None, "Transcripts are disabled for this video."
    except NoTranscriptFound:
        return None, "No transcript found for this video."
    except VideoUnavailable:
        return None, "Video is unavailable."
    except Exception as e:
        return None, f"Error fetching transcript: {str(e)}"

def get_available_transcripts(video_id):
    """Get list of available transcripts for a video"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        available_transcripts = []
        
        # Get manually created transcripts
        for transcript in transcript_list:
            available_transcripts.append({
                'language': transcript.language,
                'language_code': transcript.language_code,
                'is_generated': transcript.is_generated,
                'is_translatable': transcript.is_translatable
            })
        
        # Get generated transcripts
        for transcript in transcript_list._generated_transcripts.values():
            available_transcripts.append({
                'language': transcript.language,
                'language_code': transcript.language_code,
                'is_generated': transcript.is_generated,
                'is_translatable': transcript.is_translatable
            })
            
        return available_transcripts, None
    except Exception as e:
        return None, f"Error listing transcripts: {str(e)}"

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

def summarize_transcript(transcript, api_key, language="en", model="gpt-3.5-turbo"):
    """Summarize transcript using OpenAI API with language support"""
    # Set the API key
    openai.api_key = api_key
    
    # Check token count and split if necessary
    token_count = count_tokens(transcript)
    
    # Language-specific prompts
    language_prompts = {
        "en": "Please provide a comprehensive summary of the following YouTube video transcript. Include main points, key insights, and conclusions. Structure your response clearly:",
        "ur": "براہ کرم درج ذیل یوٹیوب ویڈیو ٹرانسکرپٹ کا جامع خلاصہ پیش کریں۔ اہم نکات، کلیدی بصیرتیں اور نتائج شامل کریں۔ اپنا جواب واضح ساخت میں پیش کریں:",
        "es": "Proporcione un resumen completo de la siguiente transcripción de video de YouTube. Incluya puntos principales, ideas clave y conclusiones. Estructure su respuesta claramente:",
        "fr": "Veuillez fournir un résumé complet de la transcription suivante de la vidéo YouTube. Incluez les points principaux, les idées clés et les conclusions. Structurez votre réponse clairement:",
        "de": "Bitte geben Sie eine umfassende Zusammenfassung des folgenden YouTube-Videotranskripts. Fügen Sie Hauptpunkte, wichtige Erkenntnisse und Schlussfolgerungen hinzu. Strukturieren Sie Ihre Antwort klar:",
        "hi": "कृपया निम्नलिखित YouTube वीडियो ट्रांसक्रिप्ट का व्यापक सारांश प्रदान करें। मुख्य बिंदु, प्रमुख अंतर्दृष्टि और निष्कर्ष शामिल करें। अपनी प्रतिक्रिया को स्पष्ट रूप से संरचित करें:",
        "ar": "يرجى تقديم ملخص شامل لنص فيديو YouTube التالي. قم بتضمين النقاط الرئيسية والرؤى الأساسية والاستنتاجات. هيكل إجابتك بوضوح:",
        "zh": "请提供以下YouTube视频转录的全面摘要。包括要点、关键见解和结论。清晰地构建您的回答:",
        "ja": "以下のYouTubeビデオの文字起こしの包括的な要約を提供してください。主なポイント、重要な洞察、結論を含めてください。回答を明確に構成してください:",
        "ru": "Предоставьте всеобъемлющее резюме следующей расшифровки видео YouTube. Включите основные моменты, ключевые идеи и выводы. Четко структурируйте свой ответ:"
    }
    
    base_prompt = language_prompts.get(language, language_prompts["en"])
    full_prompt = f"{base_prompt}\n\n{transcript}"
    
    if token_count > 3000:
        chunks = split_text(transcript)
        summaries = []
        
        for i, chunk in enumerate(chunks):
            with st.spinner(f"Summarizing part {i+1}/{len(chunks)}..."):
                chunk_prompt = f"{base_prompt}\n\n{chunk}"
                
                try:
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that summarizes YouTube video transcripts in a clear, structured manner."},
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
        final_prompt = f"Please combine these section summaries into a coherent overall summary of the video. Provide a comprehensive overview with main points and key insights:\n\n{combined_summaries}"
        
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates comprehensive video summaries."},
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
                    {"role": "system", "content": "You are a helpful assistant that summarizes YouTube video transcripts in a clear, structured manner."},
                    {"role": "user", "content": full_prompt}
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
        
        # Language selection
        st.markdown("---")
        st.markdown("### Language Preferences")
        language_options = {
            "Auto-detect": "auto",
            "English": "en",
            "Urdu": "ur",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Hindi": "hi",
            "Arabic": "ar",
            "Chinese": "zh",
            "Japanese": "ja",
            "Russian": "ru"
        }
        selected_language = st.selectbox(
            "Preferred language for transcript:",
            options=list(language_options.keys()),
            index=0
        )
        
        st.markdown("---")
        st.markdown("### How to use")
        st.markdown("""
        1. Enter your OpenAI API key (if not in secrets)
        2. Paste a YouTube URL
        3. Select language preference (optional)
        4. Click 'Generate Summary'
        5. View and copy your summary
        """)
        st.markdown("---")
        st.markdown("### Note")
        st.markdown("""
        - This tool extracts the transcript from YouTube videos
        - Supports multiple languages including English, Urdu, Spanish, etc.
        - Not all videos have available transcripts
        - Longer videos may take more time to process
        """)
    
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
        
        # Get available transcripts
        with st.spinner("Checking available transcripts..."):
            available_transcripts, error = get_available_transcripts(video_id)
            
            if error:
                st.warning(f"Could not fetch available transcripts: {error}")
                languages_to_try = None
            else:
                st.success(f"Found {len(available_transcripts)} available transcript(s)")
                if available_transcripts:
                    with st.expander("View Available Transcripts"):
                        for transcript in available_transcripts:
                            st.write(f"- {transcript['language']} ({transcript['language_code']})")
                
                # Determine which languages to try
                if selected_language != "Auto-detect":
                    languages_to_try = [language_options[selected_language]]
                else:
                    # Try English first, then other languages
                    languages_to_try = ['en', 'ur', 'es', 'fr', 'de', 'hi', 'ar', 'zh', 'ja', 'ru']
        
        with st.spinner("Fetching transcript..."):
            transcript, transcript_language = get_transcript(video_id, languages=languages_to_try)
            
        if not transcript:
            st.error(f"Could not retrieve transcript for this video. {transcript_language}")
            return
            
        st.success(f"Successfully retrieved transcript in {transcript_language if isinstance(transcript_language, str) else 'detected language'}")
            
        with st.expander("View Raw Transcript"):
            st.text(transcript[:1000] + "..." if len(transcript) > 1000 else transcript)
            
        with st.spinner("Generating summary (this may take a while for longer videos)..."):
            summary = summarize_transcript(transcript, api_key, language=language_options.get(selected_language, "en"))
            
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

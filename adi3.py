import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os, ast, PyPDF2, tempfile, re
import time
from urllib.parse import urlparse
import logging
from google.api_core.exceptions import ResourceExhausted
import uuid
from datetime import datetime
import random
import threading
import io
import atexit
import weakref

# === 🎤🔊 VOICE LIBRARIES (WITHOUT WHISPER) ===
VOICE_AVAILABLE = False
TTS_ENGINE = None
MIC_RECORDER_AVAILABLE = False

# Try voice recognition options (excluding Whisper)
try:
    # Option 1: Advanced Streamlit Mic Recorder
    from streamlit_mic_recorder import mic_recorder, speech_to_text
    MIC_RECORDER_AVAILABLE = True
    print("✅ Advanced mic recorder available")
except ImportError:
    print("📦 streamlit-mic-recorder not available")

try:
    # Option 2: Standard voice libraries
    import speech_recognition as sr
    import pyttsx3
    from audio_recorder_streamlit import audio_recorder
    import numpy as np
    from pydub import AudioSegment
    
    VOICE_AVAILABLE = True
    print("✅ Standard voice libraries available")
    
    # Initialize TTS engine with IMPROVED error handling
    try:
        TTS_ENGINE = pyttsx3.init()
        # Configure TTS
        voices = TTS_ENGINE.getProperty('voices')
        if voices:
            # Try to use female voice if available
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower() or 'hazel' in voice.name.lower():
                    TTS_ENGINE.setProperty('voice', voice.id)
                    break
        TTS_ENGINE.setProperty('rate', 160)  # Speaking speed
        TTS_ENGINE.setProperty('volume', 0.8)  # Volume
        
        # Register cleanup function to prevent shutdown errors
        def cleanup_tts():
            global TTS_ENGINE
            if TTS_ENGINE:
                try:
                    TTS_ENGINE.stop()
                    TTS_ENGINE = None
                except:
                    pass
        
        atexit.register(cleanup_tts)
        
    except Exception as e:
        print(f"TTS initialization failed: {e}")
        TTS_ENGINE = None
        
except ImportError as e:
    print(f"Standard voice libraries not available: {e}")

# === 🧠 NLP LIBRARIES ===
NLP_AVAILABLE = False
try:
    import spacy
    import nltk
    from textblob import TextBlob
    import yake
    from collections import Counter
    
    # Download required NLTK data silently
    try:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        from nltk.sentiment import SentimentIntensityAnalyzer
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize, sent_tokenize
        NLP_AVAILABLE = True
    except Exception as e:
        print(f"NLP initialization error: {e}")
        
except ImportError as e:
    print(f"NLP libraries not available: {e}")

# Suppress debug messages
logging.getLogger("streamlit").setLevel(logging.WARNING)

# === 🔑 Load API Keys ===
try:
    # For Hugging Face deployment, get the key from st.secrets
    google_api_key = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("GEMINI_API_KEY not found. Please add it to your Hugging Face Space secrets.")
    st.stop()

model = "gemini-1.5-flash"

# === 🎤🔊 SIMPLIFIED VOICE PROCESSING CLASS (NO WHISPER) ===
class VoiceProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer() if VOICE_AVAILABLE else None
        self.microphone = sr.Microphone() if VOICE_AVAILABLE else None
        self.tts_engine = None
        self.is_listening = False
        self.is_speaking = False
        self.speech_thread = None
        self._engine_ref = None
        
        # Initialize TTS engine with proper error handling
        if VOICE_AVAILABLE and TTS_ENGINE:
            try:
                # Create a new engine instance to avoid conflicts
                self.tts_engine = pyttsx3.init()
                # Store weak reference to avoid cleanup issues
                self._engine_ref = weakref.ref(self.tts_engine)
                
                # Configure TTS
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower() or 'hazel' in voice.name.lower():
                            self.tts_engine.setProperty('voice', voice.id)
                            break
                self.tts_engine.setProperty('rate', 160)
                self.tts_engine.setProperty('volume', 0.8)
                
            except Exception as e:
                print(f"Voice processor TTS init failed: {e}")
                self.tts_engine = None
        
        if self.recognizer and self.microphone:
            # Calibrate for ambient noise
            try:
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                self.recognizer.energy_threshold = 300
                self.recognizer.pause_threshold = 0.8
                self.recognizer.phrase_threshold = 0.3
            except Exception as e:
                print(f"Microphone calibration failed: {e}")
    
    def __del__(self):
        """Proper cleanup to prevent shutdown errors"""
        try:
            self.stop_speaking()
            if self.tts_engine:
                try:
                    self.tts_engine.stop()
                except:
                    pass
                self.tts_engine = None
        except:
            pass
    
    def audio_bytes_to_text(self, audio_bytes):
        """Convert audio bytes to text using standard speech recognition"""
        if not audio_bytes or not VOICE_AVAILABLE:
            return None
        
        try:
            # Convert bytes to audio file
            audio_segment = AudioSegment.from_wav(io.BytesIO(audio_bytes))
            
            # Convert to wav format for speech recognition
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)
            
            # Use speech recognition
            with sr.AudioFile(wav_io) as source:
                audio = self.recognizer.record(source)
                
            # Try Google first, then fallback to other engines
            try:
                text = self.recognizer.recognize_google(audio)
                return text
            except sr.RequestError as e:
                print(f"Google API error: {e}")
                # Try offline recognition as last resort
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    return text
                except:
                    return None
            except sr.UnknownValueError:
                return None
            
        except Exception as e:
            print(f"Voice recognition error: {str(e)}")
            return None
    
    def speak_text(self, text):
        """Convert text to speech with IMPROVED threading and error handling"""
        if not VOICE_AVAILABLE or not self.tts_engine or self.is_speaking:
            return
        
        # Stop any existing speech
        self.stop_speaking()
        
        def _speak():
            try:
                self.is_speaking = True
                
                # Clean text for better TTS
                clean_text = re.sub(r'[*#`_~]', '', text)  # Remove markdown
                clean_text = re.sub(r'\[.*?\]', '', clean_text)  # Remove citations
                clean_text = re.sub(r'\*\*.*?\*\*', '', clean_text)  # Remove bold
                clean_text = re.sub(r'#{1,6}\s+', '', clean_text)  # Remove headers
                clean_text = re.sub(r'\n+', ' ', clean_text)  # Replace newlines with spaces
                clean_text = clean_text.strip()[:500]  # Limit length
                
                if clean_text and self.tts_engine:
                    # Use try-catch for each TTS operation
                    try:
                        self.tts_engine.say(clean_text)
                        self.tts_engine.runAndWait()
                    except Exception as tts_error:
                        print(f"TTS playback error: {tts_error}")
                        
            except Exception as e:
                print(f"TTS Error: {e}")
            finally:
                self.is_speaking = False
        
        # Run TTS in background thread with daemon flag
        try:
            self.speech_thread = threading.Thread(target=_speak, daemon=True)
            self.speech_thread.start()
        except Exception as e:
            print(f"TTS thread creation error: {e}")
            self.is_speaking = False
    
    def stop_speaking(self):
        """Stop current speech with improved error handling"""
        if self.is_speaking and self.tts_engine:
            try:
                self.tts_engine.stop()
                self.is_speaking = False
                
                # Wait for thread to finish with timeout
                if self.speech_thread and self.speech_thread.is_alive():
                    self.speech_thread.join(timeout=2)
                    
            except Exception as e:
                print(f"Stop speaking error: {e}")
                self.is_speaking = False

# Initialize voice processor
voice_processor = VoiceProcessor()

# === 🧠 BACKGROUND NLP ANALYZER ===
class BackgroundNLP:
    def __init__(self):
        self.sia = None
        self.nlp = None
        
        if NLP_AVAILABLE:
            try:
                self.sia = SentimentIntensityAnalyzer()
            except Exception as e:
                print(f"Sentiment analyzer init error: {e}")
            
            try:
                for model_name in ["en_core_web_sm", "en_core_web_md"]:
                    try:
                        self.nlp = spacy.load(model_name)
                        break
                    except:
                        continue
            except Exception as e:
                print(f"SpaCy model loading error: {e}")

    def get_sentiment(self, text):
        if not NLP_AVAILABLE:
            return None
        
        try:
            blob = TextBlob(text)
            return {
                'polarity': round(blob.sentiment.polarity, 2),
                'label': 'positive' if blob.sentiment.polarity > 0.1 else 'negative' if blob.sentiment.polarity < -0.1 else 'neutral'
            }
        except:
            return None

    def extract_key_entities(self, text):
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'PRODUCT']:
                    entities.append({'text': ent.text, 'type': ent.label_})
            return entities[:5]
        except:
            return []

    def get_keywords(self, text, max_keywords=3):
        if not NLP_AVAILABLE:
            return []
        
        try:
            if len(text) < 50:
                return []
            
            try:
                kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=max_keywords)
                keywords = kw_extractor.extract_keywords(text)
                return [kw[1] for kw in keywords]
            except:
                words = word_tokenize(text.lower())
                stop_words = set(stopwords.words('english'))
                content_words = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 3]
                word_freq = Counter(content_words)
                return [word for word, freq in word_freq.most_common(max_keywords)]
        except:
            return []

    def get_context_insights(self, text):
        if not NLP_AVAILABLE or len(text.strip()) < 10:
            return {}
        
        insights = {}
        
        sentiment = self.get_sentiment(text)
        if sentiment:
            insights['sentiment'] = sentiment
        
        entities = self.extract_key_entities(text)
        if entities:
            insights['entities'] = entities
        
        keywords = self.get_keywords(text)
        if keywords:
            insights['keywords'] = keywords
        
        return insights

# Initialize background NLP
background_nlp = BackgroundNLP()

# === ⏳ DYNAMIC LOADING MESSAGES ===
LOADING_MESSAGES = [
    "🧠 Thinking deeply about your request...",
    "📚 Consulting my knowledge base...",
    "🔍 Analyzing the context...",
    "⚡ Processing your query...",
    "🎯 Formulating the perfect response...",
    "💡 Gathering insights...",
    "🔮 Working some AI magic...",
    "📊 Crunching the data...",
    "🚀 Almost there...",
    "🎨 Crafting your answer...",
    "🧩 Connecting the dots...",
    "⚙️ Fine-tuning the response..."
]

def get_random_loading_message():
    return random.choice(LOADING_MESSAGES)

# === ENHANCED SETTINGS & PERSONALIZATION ===
def init_settings():
    """Initialize settings with proper persistence"""
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "personal_prompt": """You are Calcifer AI, a helpful and knowledgeable assistant.
You are:
- Friendly and conversational in tone
- Detailed and thorough in explanations  
- Professional yet approachable
- Always helpful and supportive
Respond in a warm, engaging manner while providing accurate and useful information.""",
            "ai_name": "Calcifer AI",
            "response_style": "Balanced",
            "show_sources": True,
            "enable_smart_context": True,
            "enable_voice_input": True,
            "enable_voice_output": True,
            "auto_speak_responses": False,
            "voice_method": "auto"  # auto, advanced, standard
        }
    
    # Initialize other session state variables
    if "personalization_open" not in st.session_state:
        st.session_state.personalization_open = False
    
    if "voice_message" not in st.session_state:
        st.session_state.voice_message = None

# === PERSONALITY TEMPLATES ===
PERSONALITY_TEMPLATES = {
    "Professional Assistant": """You are a professional AI assistant.
You are:
- Formal and business-focused in communication
- Precise and concise in responses
- Expert-level in knowledge sharing
- Direct and efficient

Provide clear, professional responses suitable for business environments.""",

    "Creative Helper": """You are a creative AI assistant.
You are:
- Imaginative and inspiring in responses
- Enthusiastic about creative projects
- Supportive of artistic endeavors
- Full of fresh ideas and perspectives

Help users explore their creativity with engaging, inspiring responses.""",

    "Technical Expert": """You are a technical AI expert.
You are:
- Highly knowledgeable in technology and programming
- Detailed in technical explanations
- Precise with code and technical solutions
- Patient with complex problem-solving

Provide thorough, accurate technical assistance with clear examples.""",

    "Friendly Tutor": """You are a friendly AI tutor.
You are:
- Patient and encouraging with learning
- Clear in explanations with examples
- Supportive of student progress
- Adaptive to different learning styles

Help users learn and understand concepts with patience and clarity.""",

    "Research Assistant": """You are a research AI assistant.
You are:
- Thorough in information gathering
- Analytical in approach
- Objective and factual
- Skilled at synthesizing information

Provide comprehensive research support with well-sourced, detailed information."""
}

# === 🧵 THREAD MANAGEMENT ===
def init_threads():
    if "threads" not in st.session_state:
        st.session_state.threads = {}
    if "current_thread" not in st.session_state:
        thread_id = str(uuid.uuid4())[:8]
        st.session_state.threads[thread_id] = {
            "name": "New Chat",
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "messages": [],
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        st.session_state.current_thread = thread_id

def create_new_thread():
    thread_id = str(uuid.uuid4())[:8]
    st.session_state.threads[thread_id] = {
        "name": "New Chat",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages": [],
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    st.session_state.current_thread = thread_id
    return thread_id

def delete_thread(thread_id):
    if len(st.session_state.threads) > 1:
        del st.session_state.threads[thread_id]
        if st.session_state.current_thread == thread_id:
            st.session_state.current_thread = list(st.session_state.threads.keys())[0]
        return True
    return False

def update_thread_name(thread_id, first_message):
    if first_message and len(first_message) > 0:
        name = first_message[:30] + "..." if len(first_message) > 30 else first_message
        st.session_state.threads[thread_id]["name"] = name

def add_message_to_thread(thread_id, role, content):
    if thread_id in st.session_state.threads:
        st.session_state.threads[thread_id]["messages"].append((role, content))
        st.session_state.threads[thread_id]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        messages = st.session_state.threads[thread_id]["messages"]
        if len(messages) == 1 and role == "user":
            update_thread_name(thread_id, content)

def get_current_messages():
    if st.session_state.current_thread in st.session_state.threads:
        return st.session_state.threads[st.session_state.current_thread]["messages"]
    return []

# === RATE LIMITING ===
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0
if "request_count" not in st.session_state:
    st.session_state.request_count = 0

def safe_llm_call(llm, prompt, max_retries=3):
    current_time = time.time()
    
    if current_time - st.session_state.last_request_time > 60:
        st.session_state.request_count = 0
        st.session_state.last_request_time = current_time
    
    if st.session_state.request_count >= 8:
        time_to_wait = 60 - (current_time - st.session_state.last_request_time)
        if time_to_wait > 0:
            time.sleep(time_to_wait)
            st.session_state.request_count = 0
            st.session_state.last_request_time = time.time()
    
    for attempt in range(max_retries):
        try:
            st.session_state.request_count += 1
            response = llm.invoke(prompt)
            return response.content
        except ResourceExhausted as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10
                time.sleep(wait_time)
                continue
            else:
                return "⚠️ API rate limit exceeded. Please wait a few minutes and try again."
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return f"Error: {str(e)}"
    
    return "Failed to get response after multiple retries."

# === FILE FUNCTIONS ===
def load_text_from_uploaded_file(uploaded_file) -> str:
    try:
        if uploaded_file.name.endswith(".txt"):
            return str(uploaded_file.read(), "utf-8")
        elif uploaded_file.name.endswith(".pdf"):
            text = ""
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        else:
            return "Unsupported file format"
    except Exception as e:
        return f"Error loading file: {str(e)}"

def detect_urls(text):
    url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+'
    return re.findall(url_pattern, text)

def check_document_relevance(user_input: str, uploaded_content: dict) -> tuple:
    user_lower = user_input.lower()
    
    document_keywords = {
        "temp_profile-of-hereandnowai.pdf": ["here and now ai", "hereandnowai", "here and now", "ruthran", "ceo", "ai institute"],
    }
    
    for filename, content in uploaded_content.items():
        if filename in document_keywords:
            for keyword in document_keywords[filename]:
                if keyword in user_lower:
                    return True, filename
        
        filename_parts = filename.lower().replace("-", " ").replace("_", " ").replace(".pdf", "").replace(".txt", "").split()
        if any(part in user_lower for part in filename_parts if len(part) > 3):
            return True, filename
    
    return False, None

def safe_embedding_call(embeddings, docs, max_retries=2):
    for attempt in range(max_retries):
        try:
            return FAISS.from_documents(docs, embeddings)
        except ResourceExhausted:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return None
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                return None
            raise e
    return None

# === ENHANCED RAG WITH VOICE AWARENESS ===
def uploaded_file_rag_tool(content: str, question: str, from_voice=False) -> str:
    try:
        if not content or len(content.strip()) < 50:
            return "No sufficient content provided"

        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
        docs = splitter.create_documents([content])

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        
        vectorstore = safe_embedding_call(embeddings, docs)
        
        if vectorstore is None:
            return fallback_content_analysis(content, question, from_voice)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        system_prompt = st.session_state.settings["personal_prompt"]
        
        # Add voice-specific instructions
        voice_context = ""
        if from_voice:
            voice_context = "\n[This was a voice query - provide a conversational, clear spoken response suitable for audio output.]"
        
        # Add smart context if enabled
        nlp_context = ""
        if st.session_state.settings.get("enable_smart_context", True):
            insights = background_nlp.get_context_insights(question)
            if insights:
                context_parts = []
                if 'sentiment' in insights:
                    context_parts.append(f"User sentiment: {insights['sentiment']['label']}")
                if 'keywords' in insights and insights['keywords']:
                    context_parts.append(f"Key topics: {', '.join(insights['keywords'][:2])}")
                if context_parts:
                    nlp_context = f"\n[Context: {'; '.join(context_parts)}]"
        
        enhanced_prompt = f"""
{system_prompt}

DOCUMENT CONTEXT:
{context}

USER QUESTION: {question}{nlp_context}{voice_context}

Analyze the document content and provide a comprehensive answer.
Adapt your response tone based on the context provided.
"""

        llm = ChatGoogleGenerativeAI(model=model, google_api_key=google_api_key, temperature=0.3)
        return safe_llm_call(llm, enhanced_prompt)

    except Exception as e:
        return fallback_content_analysis(content, question, from_voice)

def fallback_content_analysis(content: str, question: str, from_voice=False) -> str:
    try:
        system_prompt = st.session_state.settings["personal_prompt"]
        
        voice_context = ""
        if from_voice:
            voice_context = "\n[Voice query - respond conversationally for audio output.]"
        
        nlp_context = ""
        if st.session_state.settings.get("enable_smart_context", True):
            insights = background_nlp.get_context_insights(question)
            if insights and 'sentiment' in insights:
                nlp_context = f"\n[User seems {insights['sentiment']['label']}]"
        
        fallback_prompt = f"""
{system_prompt}

DOCUMENT CONTENT:
{content[:6000]}

USER QUESTION: {question}{nlp_context}{voice_context}

Provide a comprehensive answer based on your personality and the document content.
"""
        
        llm = ChatGoogleGenerativeAI(model=model, google_api_key=google_api_key, temperature=0.3)
        return safe_llm_call(llm, fallback_prompt)
    except Exception as e:
        return f"Analysis failed: {str(e)}"

def web_scrape_tool(input_str: str, from_voice=False) -> str:
    try:
        url_list = detect_urls(input_str)
        if not url_list:
            return "No valid URLs found"

        combined_content = []
        for url in url_list[:2]:
            try:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                loader = WebBaseLoader([url], requests_kwargs={"headers": headers})
                documents = loader.load()
                
                for doc in documents:
                    content = doc.page_content.strip()
                    if len(content) > 200:
                        combined_content.append(content[:3000])

            except Exception as e:
                combined_content.append(f"Could not access {url}: {str(e)[:100]}")

        if not combined_content:
            return "Could not extract content from URLs"

        full_text = "\n\n---\n\n".join(combined_content)

        system_prompt = st.session_state.settings["personal_prompt"]
        
        voice_context = ""
        if from_voice:
            voice_context = "\n[Voice query - provide clear, conversational response for audio.]"
        
        nlp_context = ""
        if st.session_state.settings.get("enable_smart_context", True):
            insights = background_nlp.get_context_insights(input_str)
            if insights:
                context_parts = []
                if 'sentiment' in insights:
                    context_parts.append(f"Query sentiment: {insights['sentiment']['label']}")
                if 'keywords' in insights and insights['keywords']:
                    context_parts.append(f"Focus areas: {', '.join(insights['keywords'][:2])}")
                if context_parts:
                    nlp_context = f"\n[Analysis Context: {'; '.join(context_parts)}]"

        enhanced_prompt = f"""
{system_prompt}

WEB CONTENT:
{full_text}

USER QUERY: {input_str}{nlp_context}{voice_context}

Analyze this web content according to your personality and provide insights.
Tailor your response based on the context provided.
"""

        llm = ChatGoogleGenerativeAI(model=model, google_api_key=google_api_key, temperature=0.2)
        return safe_llm_call(llm, enhanced_prompt)

    except Exception as e:
        return f"Web scraping error: {str(e)}"

# === 🌐 STREAMLIT UI WITH VOICE INTEGRATION ===
st.set_page_config(page_title="Calcifer AI Chat", page_icon="🤖", layout="wide")

# Initialize
init_threads()
init_settings()

# === 🧵 SIDEBAR WITH ENHANCED PERSONALIZATION ===
with st.sidebar:
    st.markdown("### 🧵 Chat Threads")
    
    if st.button("➕ New Chat", use_container_width=True):
        create_new_thread()
        st.rerun()
    
    st.markdown("---")
    
    # Thread List
    for thread_id, thread_data in st.session_state.threads.items():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            button_style = "primary" if thread_id == st.session_state.current_thread else "secondary"
            if st.button(
                f"💬 {thread_data['name']}", 
                key=f"thread_{thread_id}",
                use_container_width=True,
                type=button_style
            ):
                st.session_state.current_thread = thread_id
                st.rerun()
        
        with col2:
            if len(st.session_state.threads) > 1:
                if st.button("🗑️", key=f"del_{thread_id}", help="Delete thread"):
                    if delete_thread(thread_id):
                        st.rerun()
    
    # === 🎤 VOICE CONTROLS ===
    if VOICE_AVAILABLE or MIC_RECORDER_AVAILABLE:
        st.markdown("---")
        st.markdown("### 🎤 Voice Controls")
        
        # Stop speaking button with improved feedback
        if st.button("🔇 Stop Speaking", use_container_width=True, help="Stop current speech"):
            voice_processor.stop_speaking()
            st.success("🔇 Speech stopped!")
    
    # === 🎨 PERSONAL AI BUTTON ===
    st.markdown("---")
    if st.button("🎨 Personal AI", use_container_width=True, help="Customize AI personality and behavior"):
        st.session_state.personalization_open = not st.session_state.personalization_open
        st.rerun()

# === 🎨 ENHANCED PERSONALIZATION PANEL ===
if st.session_state.personalization_open:
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🎨 Personal AI Settings")
        
        # Use forms to prevent constant reruns
        with st.form("personalization_form"):
            # AI Name
            ai_name = st.text_input(
                "🤖 AI Name",
                value=st.session_state.settings["ai_name"],
                help="What should I call myself?"
            )
            
            # Voice Settings (simplified without Whisper)
            voice_available = VOICE_AVAILABLE or MIC_RECORDER_AVAILABLE
            if voice_available:
                st.markdown("**🎤 Voice Options:**")
                
                # Voice method selection (simplified)
                voice_options = ["🔄 Auto (Best Available)"]
                if MIC_RECORDER_AVAILABLE:
                    voice_options.append("🎯 Advanced Recorder")
                if VOICE_AVAILABLE:
                    voice_options.append("🎤 Standard Recognition")
                
                voice_method_display = st.selectbox(
                    "Voice Recognition Method:",
                    voice_options,
                    help="Choose your preferred voice recognition method"
                )
                
                # Map display to internal value
                voice_method_map = {
                    "🔄 Auto (Best Available)": "auto",
                    "🎯 Advanced Recorder": "advanced",
                    "🎤 Standard Recognition": "standard"
                }
                voice_method = voice_method_map.get(voice_method_display, "auto")
                
                enable_voice_input = st.checkbox(
                    "🎤 Voice Input",
                    value=st.session_state.settings.get("enable_voice_input", True),
                    help="Enable voice input recording"
                )
                
                enable_voice_output = st.checkbox(
                    "🔊 Voice Output",
                    value=st.session_state.settings.get("enable_voice_output", True),
                    help="Enable text-to-speech responses"
                )
                
                auto_speak_responses = st.checkbox(
                    "🗣️ Auto-speak Responses",
                    value=st.session_state.settings.get("auto_speak_responses", False),
                    help="Automatically speak AI responses",
                    disabled=not enable_voice_output
                )
            else:
                st.warning("⚠️ Install voice libraries for features")
                enable_voice_input = False
                enable_voice_output = False
                auto_speak_responses = False
                voice_method = "none"
            
            # Smart Context Toggle
            enable_smart_context = st.checkbox(
                "🧠 Smart Context",
                value=st.session_state.settings.get("enable_smart_context", True),
                help="Use NLP to understand context and adapt responses",
                disabled=not NLP_AVAILABLE
            )
            
            # Response Style
            response_style = st.selectbox(
                "🎨 Response Style",
                ["Concise", "Balanced", "Detailed"],
                index=["Concise", "Balanced", "Detailed"].index(st.session_state.settings["response_style"]),
                help="How lengthy should responses be?"
            )
            
            # Show Sources Toggle
            show_sources = st.checkbox(
                "📚 Show Sources",
                value=st.session_state.settings["show_sources"],
                help="Display source info in responses"
            )
            
            # Personality Templates
            st.markdown("**🎭 Personalities:**")
            personality_choice = st.selectbox(
                "Choose template:",
                ["Custom"] + list(PERSONALITY_TEMPLATES.keys()),
                help="Select a pre-made personality"
            )
            
            # Custom Personal Prompt
            st.markdown("**💬 Custom Prompt:**")
            if personality_choice != "Custom":
                personal_prompt = st.text_area(
                    "AI personality:",
                    value=PERSONALITY_TEMPLATES[personality_choice],
                    height=120,
                    help="This defines how your AI behaves"
                )
            else:
                personal_prompt = st.text_area(
                    "AI personality:",
                    value=st.session_state.settings["personal_prompt"],
                    height=120,
                    help="This defines how your AI behaves"
                )
            
            # Form submission buttons
            col1, col2 = st.columns(2)
            with col1:
                save_settings = st.form_submit_button("💾 Save", use_container_width=True)
            with col2:
                reset_settings = st.form_submit_button("🔄 Reset", use_container_width=True)
        
        # Handle form submissions
        if save_settings:
            # Update all settings
            st.session_state.settings.update({
                "ai_name": ai_name,
                "enable_voice_input": enable_voice_input,
                "enable_voice_output": enable_voice_output,
                "auto_speak_responses": auto_speak_responses,
                "enable_smart_context": enable_smart_context,
                "response_style": response_style,
                "show_sources": show_sources,
                "personal_prompt": personal_prompt,
                "voice_method": voice_method
            })
            
            st.success("✅ Settings saved successfully!")
            time.sleep(1)
            st.rerun()
        
        if reset_settings:
            # Reset to defaults
            st.session_state.settings = {
                "personal_prompt": PERSONALITY_TEMPLATES["Professional Assistant"],
                "ai_name": "Calcifer AI",
                "response_style": "Balanced",
                "show_sources": True,
                "enable_smart_context": True,
                "enable_voice_input": True,
                "enable_voice_output": True,
                "auto_speak_responses": False,
                "voice_method": "auto"
            }
            st.success("🔄 Settings reset to defaults!")
            time.sleep(1)
            st.rerun()

# === MAIN CHAT INTERFACE ===
col1, col2 = st.columns([4, 1])
with col1:
    current_thread_name = st.session_state.threads[st.session_state.current_thread]["name"]
    ai_name = st.session_state.settings["ai_name"]
    st.title(f"🤖 {ai_name} - {current_thread_name}")
    
    # Show status indicators (simplified)
    status_indicators = []
    if NLP_AVAILABLE and st.session_state.settings.get("enable_smart_context", True):
        status_indicators.append("🧠 Smart Context")
    
    if st.session_state.settings.get("enable_voice_input", True):
        voice_method = st.session_state.settings.get("voice_method", "auto")
        if voice_method == "advanced" and MIC_RECORDER_AVAILABLE:
            status_indicators.append("🎯 Advanced Voice")
        elif voice_method == "standard" and VOICE_AVAILABLE:
            status_indicators.append("🎤 Standard Voice")
        elif voice_method == "auto":
            if MIC_RECORDER_AVAILABLE:
                status_indicators.append("🎯 Auto-Advanced")
            elif VOICE_AVAILABLE:
                status_indicators.append("🎤 Auto-Standard")
    
    if st.session_state.settings.get("enable_voice_output", True) and VOICE_AVAILABLE:
        status_indicators.append("🔊 Voice Output")
    
    if status_indicators:
        st.caption(" • ".join(status_indicators) + " enabled")
    
with col2:
    if st.button("🔄 Clear Thread", help="Clear current thread messages"):
        st.session_state.threads[st.session_state.current_thread]["messages"] = []
        st.rerun()

# === 🎤 SIMPLIFIED VOICE INPUT SECTION (NO WHISPER) ===
voice_input_text = None
if st.session_state.settings.get("enable_voice_input", True):
    st.markdown("### 🎤 Voice Input")
    
    voice_method = st.session_state.settings.get("voice_method", "auto")
    
    # Auto-detect best available method (simplified)
    if voice_method == "auto":
        if MIC_RECORDER_AVAILABLE:
            voice_method = "advanced"
        elif VOICE_AVAILABLE:
            voice_method = "standard"
    
    # Method 1: Advanced Mic Recorder
    if voice_method == "advanced" and MIC_RECORDER_AVAILABLE:
        
        col1, col2 = st.columns([3, 1])
        with col1:
            # Use advanced speech-to-text directly
            text = speech_to_text(
                language='en',
                start_prompt="🎯 Start Recording",
                stop_prompt="🎙️ Mic",
                just_once=True,
                use_container_width=True,
                key='advanced_stt'
            )
        
        if text:
            st.success(f"🎯 **Advanced transcription:** {text}")
            st.session_state.voice_message = text
            st.rerun()
    
    # Method 2: Standard Audio Recorder
    elif voice_method == "standard" and VOICE_AVAILABLE:
        st.info("🎤 **Standard Mode**: Using audio-recorder-streamlit")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            audio_bytes = audio_recorder(
                text="🎤 Click to record voice message",
                recording_color="#e8b62c",
                neutral_color="#6aa36f",
                icon_name="microphone-lines",
                icon_size="1x",
                key="voice_recorder"
            )
        
        with col2:
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
        
        # Process voice input with fallbacks
        if audio_bytes and audio_bytes != st.session_state.get("last_audio_bytes"):
            st.session_state.last_audio_bytes = audio_bytes
            
            with st.spinner("🎤 Converting speech to text..."):
                voice_input_text = voice_processor.audio_bytes_to_text(audio_bytes)
                
            if voice_input_text:
                st.success(f"🎤 **You said:** {voice_input_text}")
                st.session_state.voice_message = voice_input_text
                st.rerun()
            else:
                st.error("❌ Could not understand the audio. Please try speaking clearly and try again.")
    
    else:
        st.warning("⚠️ No voice recognition methods available. Please install required libraries.")

# === SESSION STATE FOR UPLOADED CONTENT ===
if "uploaded_content" not in st.session_state:
    st.session_state.uploaded_content = {}

# === UPLOAD SECTION ===
with st.expander("📁 Upload Documents", expanded=False):
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files for RAG", 
        accept_multiple_files=True, 
        type=["pdf", "txt"],
        help="Upload documents to add them to your knowledge base (available across all threads)"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_content:
                with st.spinner(f"📄 Processing {uploaded_file.name}..."):
                    content = load_text_from_uploaded_file(uploaded_file)
                    st.session_state.uploaded_content[uploaded_file.name] = content
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded and processed!")

    # Show available documents
    if st.session_state.uploaded_content:
        st.markdown("**📚 Available Documents:**")
        cols = st.columns(min(3, len(st.session_state.uploaded_content)))
        for i, filename in enumerate(st.session_state.uploaded_content.keys()):
            with cols[i % len(cols)]:
                st.info(f"📄 {filename}")

# === CHAT MESSAGES DISPLAY WITH VOICE OUTPUT ===
messages = get_current_messages()
for i, (role, msg) in enumerate(messages):
    with st.chat_message(role):
        st.markdown(msg)
        
        # Add speak button for AI responses with unique keys
        if (role == "assistant" and 
            st.session_state.settings.get("enable_voice_output", True) and
            (VOICE_AVAILABLE or TTS_ENGINE)):
            speak_key = f"speak_{i}_{hash(msg[:50])}"  # Unique key
            if st.button("🔊 Speak", key=speak_key, help="Read this response aloud"):
                voice_processor.speak_text(msg)

# === USER INPUT WITH VOICE PROCESSING ===
ai_name = st.session_state.settings["ai_name"]

# Check for voice message
user_input = None
from_voice = False

if st.session_state.voice_message:
    user_input = st.session_state.voice_message
    from_voice = True
    st.session_state.voice_message = None  # Clear after use

# Regular text input
if not user_input:
    user_input = st.chat_input(f"Message {ai_name}... (Thread: {current_thread_name})")

if user_input:
    add_message_to_thread(st.session_state.current_thread, "user", user_input)

    # Show user message immediately
    with st.chat_message("user"):
        if from_voice:
            voice_method = st.session_state.settings.get("voice_method", "auto")
            voice_icon = "🎯" if voice_method == "advanced" else "🎤"
            st.markdown(f"{voice_icon} {user_input}")
        else:
            st.markdown(user_input)

    # Process input with voice-enhanced responses
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        loading_message = get_random_loading_message()
        message_placeholder.text(loading_message)
        
        try:
            start_time = time.time()
            last_message_time = start_time
            
            detected_urls = detect_urls(user_input)
            if detected_urls:
                while time.time() - start_time < 2:
                    if time.time() - last_message_time > 1.5:
                        loading_message = get_random_loading_message()
                        message_placeholder.text(loading_message)
                        last_message_time = time.time()
                    time.sleep(0.1)
                
                result = web_scrape_tool(user_input, from_voice)
                message_placeholder.empty()
                response = f"**🌐 Web Analysis:**\n\n{result}"
                st.markdown(response)
                add_message_to_thread(st.session_state.current_thread, "assistant", response)
                
                # Auto-speak if enabled
                if (from_voice and 
                    st.session_state.settings.get("enable_voice_output", True) and
                    st.session_state.settings.get("auto_speak_responses", False)):
                    voice_processor.speak_text(result)
            
            elif st.session_state.uploaded_content:
                is_relevant, relevant_file = check_document_relevance(user_input, st.session_state.uploaded_content)
                
                if is_relevant and relevant_file:
                    while time.time() - start_time < 2:
                        if time.time() - last_message_time > 1.5:
                            loading_message = get_random_loading_message()
                            message_placeholder.text(loading_message)
                            last_message_time = time.time()
                        time.sleep(0.1)
                    
                    content = st.session_state.uploaded_content[relevant_file]
                    result = uploaded_file_rag_tool(content, user_input, from_voice)
                    message_placeholder.empty()
                    source_info = f"\n\n*📄 Source: {relevant_file}*" if st.session_state.settings["show_sources"] else ""
                    response = f"**📄 Document Analysis:**\n\n{result}{source_info}"
                    st.markdown(response)
                    add_message_to_thread(st.session_state.current_thread, "assistant", response)
                    
                    # Auto-speak if enabled
                    if (from_voice and 
                        st.session_state.settings.get("enable_voice_output", True) and
                        st.session_state.settings.get("auto_speak_responses", False)):
                        voice_processor.speak_text(result)
                
                elif any(word in user_input.lower() for word in ["what", "how", "explain", "tell", "describe", "summarize", "analyze", "who", "ceo"]):
                    while time.time() - start_time < 2:
                        if time.time() - last_message_time > 1.5:
                            loading_message = get_random_loading_message()
                            message_placeholder.text(loading_message)
                            last_message_time = time.time()
                        time.sleep(0.1)
                    
                    first_file = list(st.session_state.uploaded_content.keys())[0]
                    content = st.session_state.uploaded_content[first_file]
                    result = uploaded_file_rag_tool(content, user_input, from_voice)
                    message_placeholder.empty()
                    source_info = f"\n\n*📚 Source: {first_file}*" if st.session_state.settings["show_sources"] else ""
                    response = f"**📚 Knowledge Base:**\n\n{result}{source_info}"
                    st.markdown(response)
                    add_message_to_thread(st.session_state.current_thread, "assistant", response)
                    
                    # Auto-speak if enabled
                    if (from_voice and 
                        st.session_state.settings.get("enable_voice_output", True) and
                        st.session_state.settings.get("auto_speak_responses", False)):
                        voice_processor.speak_text(result)
                
                else:
                    # Regular chat with documents available
                    while time.time() - start_time < 1.5:
                        if time.time() - last_message_time > 1:
                            loading_message = get_random_loading_message()
                            message_placeholder.text(loading_message)
                            last_message_time = time.time()
                        time.sleep(0.1)
                    
                    # Build enhanced prompt with voice context
                    system_prompt = st.session_state.settings["personal_prompt"]
                    style_instruction = {
                        "Concise": "Be brief and to the point.",
                        "Balanced": "Provide a good balance of detail and brevity.",
                        "Detailed": "Be comprehensive and thorough in your explanations."
                    }[st.session_state.settings["response_style"]]
                    
                    # Voice-specific context
                    voice_context = ""
                    if from_voice:
                        voice_context = "\n[Voice input - respond conversationally for potential audio output.]"
                    
                    # Add NLP context awareness
                    nlp_context = ""
                    if st.session_state.settings.get("enable_smart_context", True):
                        insights = background_nlp.get_context_insights(user_input)
                        if insights:
                            context_parts = []
                            if 'sentiment' in insights:
                                sentiment_info = insights['sentiment']
                                if sentiment_info['label'] == 'negative':
                                    context_parts.append("User seems frustrated or negative - be extra supportive and helpful")
                                elif sentiment_info['label'] == 'positive':
                                    context_parts.append("User seems positive - match their enthusiasm")
                            
                            if 'entities' in insights and insights['entities']:
                                entity_names = [ent['text'] for ent in insights['entities'][:2]]
                                context_parts.append(f"Key topics mentioned: {', '.join(entity_names)}")
                            
                            if 'keywords' in insights and insights['keywords']:
                                context_parts.append(f"Focus on: {', '.join(insights['keywords'][:2])}")
                            
                            if context_parts:
                                nlp_context = f"\n[Context guidance: {'; '.join(context_parts)}]"
                    
                    enhanced_prompt = f"""
{system_prompt}

Style: {style_instruction}

USER QUESTION: {user_input}{nlp_context}{voice_context}

Respond according to your defined personality and style.
Use the context guidance to adapt your tone and focus appropriately.
"""
                    
                    llm = ChatGoogleGenerativeAI(model=model, google_api_key=google_api_key, temperature=0.4)
                    result = safe_llm_call(llm, enhanced_prompt)
                    message_placeholder.empty()
                    
                    if st.session_state.uploaded_content and st.session_state.settings["show_sources"]:
                        file_list = ", ".join(st.session_state.uploaded_content.keys())
                        result += f"\n\n*💡 Available documents: {file_list}*"
                    
                    st.markdown(result)
                    add_message_to_thread(st.session_state.current_thread, "assistant", result)
                    
                    # Auto-speak if enabled
                    if (from_voice and 
                        st.session_state.settings.get("enable_voice_output", True) and
                        st.session_state.settings.get("auto_speak_responses", False)):
                        voice_processor.speak_text(result)
            
            else:
                # Regular chat without documents
                while time.time() - start_time < 1.5:
                    if time.time() - last_message_time > 1:
                        loading_message = get_random_loading_message()
                        message_placeholder.text(loading_message)
                        last_message_time = time.time()
                    time.sleep(0.1)
                
                # Enhanced regular chat with voice context
                system_prompt = st.session_state.settings["personal_prompt"]
                style_instruction = {
                    "Concise": "Be brief and to the point.",
                    "Balanced": "Provide a good balance of detail and brevity.", 
                    "Detailed": "Be comprehensive and thorough in your explanations."
                }[st.session_state.settings["response_style"]]
                
                # Voice-specific context
                voice_context = ""
                if from_voice:
                    voice_context = "\n[Voice input - provide clear, conversational response.]"
                
                # Add NLP context for regular chat
                nlp_context = ""
                if st.session_state.settings.get("enable_smart_context", True):
                    insights = background_nlp.get_context_insights(user_input)
                    if insights:
                        context_parts = []
                        if 'sentiment' in insights:
                            sentiment_info = insights['sentiment']
                            if sentiment_info['label'] == 'negative':
                                context_parts.append("Be supportive and solution-focused")
                            elif sentiment_info['label'] == 'positive':
                                context_parts.append("Match the user's positive energy")
                        
                        if 'keywords' in insights and insights['keywords']:
                            context_parts.append(f"Key topics: {', '.join(insights['keywords'][:2])}")
                        
                        if context_parts:
                            nlp_context = f"\n[Response guidance: {'; '.join(context_parts)}]"
                
                enhanced_prompt = f"""
{system_prompt}

Style: {style_instruction}

USER QUESTION: {user_input}{nlp_context}{voice_context}

Respond according to your defined personality and style.
Adapt your tone and approach based on the guidance provided.
"""
                
                llm = ChatGoogleGenerativeAI(model=model, google_api_key=google_api_key, temperature=0.4)
                result = safe_llm_call(llm, enhanced_prompt)
                message_placeholder.empty()
                st.markdown(result)
                add_message_to_thread(st.session_state.current_thread, "assistant", result)
                
                # Auto-speak if enabled
                if (from_voice and 
                    st.session_state.settings.get("enable_voice_output", True) and
                    st.session_state.settings.get("auto_speak_responses", False)):
                    voice_processor.speak_text(result)

        except Exception as e:
            message_placeholder.empty()
            error_msg = f"❌ Something went wrong: {str(e)}"
            st.error(error_msg)
            add_message_to_thread(st.session_state.current_thread, "assistant", error_msg)

    st.rerun()

# === 📚 SIMPLIFIED INSTALLATION GUIDE ===
any_voice_available = VOICE_AVAILABLE or MIC_RECORDER_AVAILABLE

if not any_voice_available:
    with st.expander("🚀 Enable Voice Features", expanded=False):
        st.markdown("""
        **🎯 Option 1: Advanced Streamlit Mic Recorder**
        
        ```
        pip install streamlit-mic-recorder
        ```
        
        **🎤 Option 2: Standard Voice Libraries**
        
        ```
        pip install SpeechRecognition pyttsx3 pyaudio audio-recorder-streamlit pydub
        ```
        
        **System Dependencies:**
        - **Windows**: Usually works out of the box
        - **Mac**: `brew install portaudio`  
        - **Linux**: `sudo apt install portaudio19-dev python3-pyaudio espeak espeak-data`
        
        **🌟 What you'll get:**
        - 🎯 **Advanced Recorder**: Real-time transcription, browser-based recording
        - 🎤 **Standard Voice**: Reliable Google Speech Recognition fallback
        - 🔊 **Text-to-Speech**: Natural voice responses with multiple voice options
        - 🗣️ **Auto-speak**: Hands-free interaction for voice queries
        - 🧠 **Voice-aware AI**: Adapts responses specifically for audio output
        """)

if not NLP_AVAILABLE:
    with st.expander("🧠 Enable Smart Context Features", expanded=False):
        st.markdown("""
        **To enable smart context analysis:**
        
        ```
        pip install spacy nltk textblob yake
        python -m spacy download en_core_web_sm
        ```
        
        **🌟 Features you'll get:**
        - 🧠 **Sentiment analysis** - AI understands your mood and adapts responses
        - 🎯 **Keyword extraction** - AI focuses on the most important topics  
        - 🏷️ **Entity recognition** - Identifies people, places, organizations automatically
        - 📊 **Context-aware responses** - Much smarter, more relevant answers
        - 🎨 **Adaptive personality** - AI changes tone based on your emotional state
        """)
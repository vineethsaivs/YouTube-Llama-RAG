import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
import tiktoken
from sentence_transformers import SentenceTransformer, util
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file (if needed for other settings)
load_dotenv()

# Initialize the SentenceTransformer model
model = SentenceTransformer("bert-base-nli-mean-tokens")

# ----- Utility Functions ----- #
def extract_video_id(video_input):
    """Extract the YouTube video ID from a URL or return the input if it's already an ID."""
    match = re.search(r'(?:v=|youtu\.be/)([^&?]+)', video_input)
    if match:
        return match.group(1)
    return video_input.strip()

def call_llama_api(messages, max_tokens=1000, temperature=0.5):
    """
    Convert a list of messages into a prompt and call the local Ollama model.
    This function calls: `ollama run llama3.1:8b`
    """
    # Build a prompt from the messages.
    prompt = ""
    for message in messages:
        role = message.get("role", "").lower()
        content = message.get("content", "")
        if role == "system":
            prompt += "System: " + content + "\n"
        elif role == "user":
            prompt += "User: " + content + "\n"
        elif role == "assistant":
            prompt += "Assistant: " + content + "\n"
        else:
            prompt += content + "\n"
    
    # Optionally, you could add instructions to control max_tokens or temperature
    # if your Ollama setup supports them via the prompt.

    # Call Ollama using subprocess
    result = subprocess.run(
        ["ollama", "run", "llama3.1:8b"],
        input=prompt,
        text=True,
        capture_output=True
    )
    if result.returncode != 0:
        raise Exception("Ollama run failed: " + result.stderr)
    output_text = result.stdout.strip()
    # Mimic the structure returned by the previous API call
    return {"choices": [{"message": {"content": output_text}}]}

def split_into_chunks(text, tokens=500):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    words = encoding.encode(text)
    chunks = []
    for i in range(0, len(words), tokens):
        chunks.append(" ".join(encoding.decode(words[i : i + tokens])))
    return chunks

def clean_transcript(transcript_text):
    """
    Split transcript text into chunks and use Ollama to clean/group the transcript.
    Returns a list of cleaned paragraphs.
    """
    chunks = split_into_chunks(transcript_text)
    st.info(f"Split transcript into {len(chunks)} chunks...")
    
    def process_chunk(chunk):
        messages = [
            {
                "role": "system",
                "content": (
                    "You will be given the full text transcript of a YouTube video. "
                    "Clean the data by splitting the transcript into individual sentences while maintaining correct grammar, "
                    "and then group 1-3 similar sentences together into paragraphs. Return the cleaned paragraphs."
                )
            },
            {"role": "user", "content": f"YOUR DATA TO PASS IN: {chunk}"}
        ]
        result = call_llama_api(messages, max_tokens=1000, temperature=0.5)
        return result["choices"][0]["message"]["content"].strip()
    
    with ThreadPoolExecutor() as executor:
        cleaned_chunks = list(executor.map(process_chunk, chunks))
    
    # Split the cleaned output by newlines and combine into a list of paragraphs
    paragraphs = []
    for text in cleaned_chunks:
        paragraphs.extend([line.strip() for line in text.splitlines() if line.strip()])
    return paragraphs

def get_transcript(video_id):
    transcript_dict = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = " ".join([t["text"] for t in transcript_dict])
    return transcript

def generate_response(video_input, query):
    video_id = extract_video_id(video_input)
    
    # Retrieve transcript
    transcript_text = get_transcript(video_id)
    
    # Clean transcript using Ollama
    cleaned_paragraphs = clean_transcript(transcript_text)
    
    # Create embeddings for cleaned paragraphs
    embeddings = model.encode(cleaned_paragraphs)
    # Create embedding for the query
    query_embedding = model.encode(query)
    hits = util.semantic_search(query_embedding, embeddings, top_k=3)[0]
    best_hit_index = hits[0]['corpus_id']
    context = cleaned_paragraphs[best_hit_index]
    
    # Create final prompt messages
    messages = [
        {
            "role": "system",
            "content": (
                "You will be given a query along with relevant context extracted from a YouTube video transcript. "
                "Generate a context-aware, concise, and conversational response."
            )
        },
        {
            "role": "user",
            "content": f"QUERY: {query}\nCONTEXT: {context}"
        }
    ]
    result = call_llama_api(messages, max_tokens=1500, temperature=0.5)
    final_response = result["choices"][0]["message"]["content"].strip()
    return transcript_text, cleaned_paragraphs, final_response

# ----- Streamlit App ----- #
st.title("YouTube RAG with Ollama's Llama 3.1 (8B)")
st.write("Enter a YouTube video URL (or ID) and your query. The app retrieves the transcript, cleans it, performs semantic search, and then generates a context-aware answer using your local Llama 3.1 8B model via Ollama.")

video_input = st.text_input("YouTube Video URL or ID")
query = st.text_input("Enter your query")

if st.button("Generate Response"):
    if video_input and query:
        with st.spinner("Processing..."):
            try:
                transcript_text, paragraphs, response_text = generate_response(video_input, query)
                st.success("Response generated!")
                st.subheader("Final Response")
                st.write(response_text)
                
                st.subheader("Transcript Summary")
                for i, para in enumerate(paragraphs, start=1):
                    st.markdown(f"**Paragraph {i}:** {para}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please provide both a YouTube video URL (or ID) and a query.")

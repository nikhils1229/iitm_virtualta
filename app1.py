import os
import json
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import chromadb
from chromadb.config import Settings
import traceback
from datetime import datetime
import requests # NEW: Import requests for web API calls

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize ChromaDB (in-memory for simplicity)
chroma_client = chromadb.Client()

# --- NEW: Brave Search API Key ---
BRAVE_API_KEY = os.getenv('BRAVE_API_KEY')
if not BRAVE_API_KEY:
    print("WARNING: BRAVE_API_KEY environment variable not set. Web search functionality will not work.")

# Create sample TDS knowledge base
def initialize_knowledge_base():
    """Initialize sample TDS course data"""
    try:
        # Try to get existing collection, if not create new one
        collection = chroma_client.get_collection("tds_knowledge")
    except: # This broad exception handling is fine for simple setup, but in production, be more specific
        collection = chroma_client.create_collection("tds_knowledge")

        # Sample TDS course content based on research
        sample_data = [
            {
                "content": "You must use gpt-3.5-turbo-0125 model as specified in the assignment, even if AI proxy only supports gpt-4o-mini. Use OpenAI API directly for accurate results.",
                "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939",
                "text": "Use the model mentioned in the question"
            },
            {
                "content": "If a student scores 10/10 on GA4 plus bonus points, the dashboard will show 110 (10 base points + 10 bonus = 110% displayed).",
                "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga4-data-sourcing-discussion-thread-tds-jan-2025/165959",
                "text": "Dashboard shows bonus scores as percentages over 100"
            },
            {
                "content": "For TDS course, Podman is recommended over Docker for better security and rootless containers. However, Docker is acceptable if you're already familiar with it.",
                "url": "https://tds.s-anand.net/#/docker",
                "text": "Podman vs Docker guidance for TDS course"
            },
            {
                "content": "Tools in Data Science covers development tools (uv, git, bash, sqlite), deployment tools (Docker, Vercel, ngrok, FastAPI), LLMs (prompt engineering, RAG, embeddings), data sourcing, preparation, analysis and visualization.",
                "url": "https://tds.s-anand.net/",
                "text": "TDS course curriculum overview"
            },
            {
                "content": "Course includes 7 graded assignments, 2 projects, and a remote online exam. It's a fairly tough course with high failure rates.",
                "url": "https://github.com/sanand0/tools-in-data-science-public",
                "text": "TDS course structure and difficulty"
            },
            {
                "content": "For cost calculations with GPT models: gpt-3.5-turbo-0125 costs $0.0015 per 1K input tokens and $0.002 per 1K output tokens. Use tokenizer to count tokens and multiply by given rate.",
                "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939",
                "text": "Token cost calculation method"
            }
        ]

        # Add documents to collection
        for i, item in enumerate(sample_data):
            collection.add(
                documents=[item["content"]],
                metadatas=[{"url": item["url"], "text": item["text"]}],
                ids=[f"doc_{i}"]
            )

    return collection

# Initialize knowledge base
knowledge_base = initialize_knowledge_base()

def search_knowledge_base(query, n_results=3):
    """Search the knowledge base for relevant information"""
    try:
        # ChromaDB query now includes distances
        results = knowledge_base.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances'] # NEW: Include distances for potential confidence check
        )

        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            docs = results['documents'][0]
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            distances = results['distances'][0] if results['distances'] else [] # NEW

            for i, doc in enumerate(docs):
                metadata = metadatas[i] if i < len(metadatas) else {}
                distance = distances[i] if i < len(distances) else None # NEW
                formatted_results.append({
                    "content": doc,
                    "url": metadata.get("url", ""),
                    "text": metadata.get("text", ""),
                    "distance": distance # NEW: Include distance
                })

        return formatted_results
    except Exception as e:
        print(f"Search error: {e}")
        return []

# --- NEW FUNCTION: perform_web_search ---
def perform_web_search(query, num_results=3):
    """Performs a web search using Brave Search API."""
    if not BRAVE_API_KEY:
        print("Brave API key not set. Skipping web search.")
        return None

    headers = {
        "X-Subscription-Token": BRAVE_API_KEY,
        "Accept": "application/json"
    }
    params = {
        "q": query,
        "count": num_results,
        "country": "us" # You can customize country code (e.g., 'in' for India)
    }
    url = "https://api.brave.com/serp/v1/search"

    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        search_data = response.json()

        web_results_text = []
        if search_data and 'web_results' in search_data:
            for i, result in enumerate(search_data['web_results'][:num_results]):
                title = result.get('title', 'No Title')
                url = result.get('url', '#')
                snippet = result.get('description', 'No snippet available.')
                web_results_text.append(f"Result {i+1}:\nTitle: {title}\nURL: {url}\nSnippet: {snippet}\n")
            return "\n".join(web_results_text)
        return None

    except requests.exceptions.RequestException as e:
        print(f"Brave Search API request failed: {e}")
        return None
    except Exception as e:
        print(f"Error parsing Brave Search results: {e}")
        return None

def analyze_image(base64_image, question):
    """Analyze image using GPT-4o-mini vision capabilities"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Question: {question}\n\nPlease analyze this image and provide relevant information to answer the question."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Image analysis error: {e}")
        return "Could not analyze the provided image."

# --- MODIFIED: generate_answer to accept web_search_results ---
def generate_answer(question, context, image_analysis=None, web_search_results=None):
    """Generate answer using GPT-3.5-turbo-0125"""
    try:
        # Build context from internal search results
        context_text = "\n\n".join([f"- {item['content']}" for item in context])

        # NEW: Add web search results to context if available
        web_context_text = ""
        if web_search_results:
            web_context_text = f"\n\nAdditional information from web search:\n{web_search_results}"

        prompt = f"""You are a helpful teaching assistant for the Tools in Data Science (TDS) course at IIT Madras.

Question: {question}

Context from course materials:
{context_text}

{f"Image analysis: {image_analysis}" if image_analysis else ""}
{web_context_text}

Please provide a helpful and accurate answer.
**Prioritize information from the 'Context from course materials' first.**
**If the 'Context from course materials' is insufficient or the question is general knowledge, then use the 'Additional information from web search'. If you use information from web search, clearly state that the information is from an external source.**

If the question is not related to TDS course content and cannot be answered by either the course materials or web search results, politely explain that you can only help with TDS-related questions or indicate if it's outside the scope of your current knowledge.

Answer format should be clear and concise."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500, # Increased max_tokens to accommodate longer answers with web search
            temperature=0.1
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Answer generation error: {e}")
        return "I apologize, but I'm having trouble generating an answer right now. Please try again later."

@app.route('/')
def health_check():
    return jsonify({"status": "TDS Virtual TA is running", "timestamp": datetime.now().isoformat()})

@app.route('/api/', methods=['POST'])
def answer_question():
    try:
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400

        question = data['question']
        image_data = data.get('image')

        # --- MODIFIED LOGIC: Prioritize internal knowledge, then web search ---
        search_results = search_knowledge_base(question, n_results=3) # Keep n_results for internal search

        web_search_results_text = None
        # Simple check: If no relevant internal documents are found (empty search_results list)
        # you could also add a check based on 'distance' if you wanted a confidence threshold.
        if not search_results:
            print(f"No relevant internal results found for '{question}'. Attempting web search...")
            web_search_results_text = perform_web_search(question)
            if not web_search_results_text:
                print(f"No significant web search results found for '{question}'.")
            else:
                print("Web search successful.")
        else:
            print(f"Relevant internal results found for '{question}'. Skipping web search.")


        # Analyze image if provided
        image_analysis = None
        if image_data:
            try:
                # Remove data URL prefix if present
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                image_analysis = analyze_image(image_data, question)
            except Exception as e:
                print(f"Image processing error: {e}")

        # Generate answer - NEW: Pass web_search_results_text
        answer = generate_answer(question, search_results, image_analysis, web_search_results_text)

        # Format links - Only include links from internal knowledge base for now
        # You might extend this to include URLs from web_search_results if needed.
        links = []
        for result in search_results:
            if result['url'] and result['text']:
                links.append({
                    "url": result['url'],
                    "text": result['text']
                })

        # NEW: Adjust links based on where the answer came from
        if not links and web_search_results_text: # If answer likely came from web search
             links.append({
                "url": f"https://www.google.com/search?q={question.replace(' ', '+')}",
                "text": "Google Search for more info"
            })
        elif not links and not web_search_results_text: # If no info found at all
            links.append({
                "url": "https://tds.s-anand.net/",
                "text": "TDS Course Materials"
            })
        elif not links and search_results: # Fallback to general course link if internal results had no specific links
             links.append({
                "url": "https://tds.s-anand.net/",
                "text": "TDS Course Materials"
            })


        response = {
            "answer": answer,
            "links": links
        }

        return jsonify(response)

    except Exception as e:
        print(f"API error: {e}")
        print(traceback.format_exc())
        return jsonify({
            "answer": "I apologize, but I encountered an error while processing your question. Please try again.",
            "links": [{
                "url": "https://tds.s-anand.net/",
                "text": "TDS Course Materials"
            }]
        }), 500

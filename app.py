
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

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize ChromaDB (in-memory for simplicity)
chroma_client = chromadb.Client()

# Create sample TDS knowledge base
def initialize_knowledge_base():
    """Initialize sample TDS course data"""
    try:
        # Try to get existing collection, if not create new one
        collection = chroma_client.get_collection("tds_knowledge")
    except:
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
        results = knowledge_base.query(
            query_texts=[query],
            n_results=n_results
        )

        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            docs = results['documents'][0]
            metadatas = results['metadatas'][0] if results['metadatas'] else []

            for i, doc in enumerate(docs):
                metadata = metadatas[i] if i < len(metadatas) else {}
                formatted_results.append({
                    "content": doc,
                    "url": metadata.get("url", ""),
                    "text": metadata.get("text", "")
                })

        return formatted_results
    except Exception as e:
        print(f"Search error: {e}")
        return []

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

def generate_answer(question, context, image_analysis=None):
    """Generate answer using GPT-3.5-turbo-0125"""
    try:
        # Build context from search results
        context_text = "\n\n".join([f"- {item['content']}" for item in context])

        prompt = f"""You are a helpful teaching assistant for the Tools in Data Science (TDS) course at IIT Madras. 

Question: {question}

Context from course materials:
{context_text}

{f"Image analysis: {image_analysis}" if image_analysis else ""}

Please provide a helpful and accurate answer based on the course context. If the question is not related to TDS course content, politely explain that you can only help with TDS-related questions.

Answer format should be clear and concise."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
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

        # Search knowledge base
        search_results = search_knowledge_base(question)

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

        # Generate answer
        answer = generate_answer(question, search_results, image_analysis)

        # Format links
        links = []
        for result in search_results:
            if result['url'] and result['text']:
                links.append({
                    "url": result['url'],
                    "text": result['text']
                })

        # Ensure we have at least some links
        if not links and search_results:
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

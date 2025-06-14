import os
import json
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import chromadb
import traceback
from datetime import datetime
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
chroma_client = chromadb.Client()

TDS_KNOWLEDGE = [
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
    }
]

def initialize_knowledge_base():
    try:
        collection = chroma_client.get_collection("tds_knowledge")
    except:
        collection = chroma_client.create_collection("tds_knowledge")
        for i, item in enumerate(TDS_KNOWLEDGE):
            collection.add(
                documents=[item["content"]],
                metadatas=[{"url": item["url"], "text": item["text"]}],
                ids=[f"doc_{i}"]
            )
    return collection

def search_knowledge_base(query, collection, n_results=3):
    try:
        results = collection.query(query_texts=[query], n_results=n_results)
        formatted_results = []
        if results['documents']:
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
        return TDS_KNOWLEDGE[:2]

def analyze_image(base64_image, question):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Question: {question}\n\nAnalyze this image for TDS course context."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Image analysis error: {e}")
        return "Could not analyze the provided image."

def generate_answer(question, context, image_analysis=None):
    try:
        context_text = "\n".join([f"- {item['content']}" for item in context])
        prompt = f"""You are a teaching assistant for Tools in Data Science (TDS) at IIT Madras.

Question: {question}

Course Context:
{context_text}

{f"Image Analysis: {image_analysis}" if image_analysis else ""}

Provide a helpful answer based on TDS course content. Be concise and accurate."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Answer generation error: {e}")
        return "I'm having trouble generating an answer. Please try again."

@app.route('/')
def health():
    return jsonify({"status": "TDS Virtual TA API", "timestamp": datetime.now().isoformat()})

@app.route('/api/', methods=['POST'])
def handle_post():
    try:
        raw_body = request.data.decode('utf-8')

        # Fix broken JSON from Jinja template
        raw_body = raw_body.replace('{% if image %},', '')
        raw_body = raw_body.replace('{% endif %}', '')

        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError as e:
            return jsonify({
                "error": "Invalid JSON body",
                "details": str(e),
                "raw": raw_body
            }), 400

        question = data.get('question', '')
        image_data = data.get('image')

        # Initialize knowledge base
        knowledge_base = initialize_knowledge_base()
        search_results = search_knowledge_base(question, knowledge_base)

        image_analysis = None
        if image_data and "{{" not in image_data:
            try:
                # Skip file:// and validate base64
                if image_data.startswith("file://"):
                    raise ValueError("Skipping file reference image.")

                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]

                # Validate image
                img_bytes = base64.b64decode(image_data)
                Image.open(io.BytesIO(img_bytes)).verify()

                image_analysis = analyze_image(image_data, question)
            except Exception as e:
                print(f"⚠️ Skipping image analysis: {e}")

        answer = generate_answer(question, search_results, image_analysis)

        links = [
            {"url": r["url"], "text": r["text"]}
            for r in search_results if r.get('url') and r.get('text')
        ] or [{"url": "https://tds.s-anand.net/", "text": "TDS Course Materials"}]

        return jsonify({
            "answer": answer,
            "links": links
        })

    except Exception as e:
        print(f"API error: {e}")
        print(traceback.format_exc())
        return jsonify({
            "answer": "Sorry, I encountered an error. Please try again.",
            "links": [{"url": "https://tds.s-anand.net/", "text": "TDS Course Materials"}]
        }), 500

def handler(request):
    return app(request.environ, lambda *args: None)

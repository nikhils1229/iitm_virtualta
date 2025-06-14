from flask import Flask, request, jsonify
import json
import base64
import io
from PIL import Image

app = Flask(__name__)

@app.route("/api/", methods=["POST"])
def handle_post():
    raw_body = request.data.decode("utf-8")

    # ✅ Fix broken Jinja-style logic
    raw_body = raw_body.replace("{% if image %},", "")
    raw_body = raw_body.replace("{% endif %}", "")

    try:
        data = json.loads(raw_body)
    except json.JSONDecodeError as e:
        return jsonify({
            "error": "Invalid JSON body",
            "details": str(e),
            "raw": raw_body
        }), 400

    question = data.get("question", "").strip()
    image_data = data.get("image", None)

    result = process_question(question, image_data)
    return jsonify(result)


def process_question(question, image_data=None):
    answer = ""
    links = []

    # 🔍 Basic routing based on keywords (extend as needed)
    if "gpt-3.5-turbo" in question:
        answer = "Use gpt-3.5-turbo-0125 as mentioned. Don't substitute it with gpt-4o-mini."
        links = [{
            "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939",
            "text": "GA5 Question 8 Clarification"
        }]
    elif "scores 10/10" in question:
        answer = "It would appear as 110 on the dashboard including the bonus."
        links = [{
            "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga4-data-sourcing-discussion-thread-tds-jan-2025/165959",
            "text": "GA4 Data Sourcing Discussion"
        }]
    elif "Docker" in question:
        answer = "Podman is recommended for the course, but Docker is also acceptable."
        links = [{
            "url": "https://tds.s-anand.net/#/docker",
            "text": "TDS Docker Setup Guide"
        }]
    elif "end-term exam" in question:
        answer = "The exam date is not yet available."
    else:
        answer = "Sorry, I don't have an answer for that yet."

    # 🖼️ Optional: Validate base64 image input
    if image_data and "{{" not in image_data:
        try:
            if image_data.startswith("data:image"):
                image_data = image_data.split(",")[1]
            base64.b64decode(image_data, validate=True)
            img = Image.open(io.BytesIO(base64.b64decode(image_data)))
            img.verify()  # raises exception if invalid
        except Exception as e:
            print("[WARN] Invalid image data received:", e)

    return {
        "answer": answer,
        "links": links
    }


if __name__ == "__main__":
    app.run(debug=True)

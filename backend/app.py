import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-3.5-flash-lite"


@app.route("/")
def home():
    return "My AI Bot backend is running!"


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON data received"}), 400

    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400

    try:

        response = client.models.generate_content(
            model=MODEL,
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a helpful AI assistant. "
                    "Be clear, friendly and useful. "
                    "Do not pretend to be human."
                ),
                temperature=0.7,
                max_output_tokens=1000
            )
        )

        return jsonify({
            "reply": response.text
        })

    except Exception as error:

        print(error)

        return jsonify({
            "error": "The AI could not generate a response."
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )

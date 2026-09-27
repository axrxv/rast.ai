import os
import base64
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS

from google import genai
from google.genai import types

from huggingface_hub import InferenceClient


app = Flask(__name__)
CORS(app)


# =========================================================
# GEMINI SETUP
# =========================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

GEMINI_MODEL = "gemini-3.5-flash-lite"


# =========================================================
# HUGGING FACE IMAGE SETUP
# =========================================================

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set.")

image_client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN
)

IMAGE_MODEL = "black-forest-labs/FLUX.1-dev"


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    return "rast.ai backend is running."


# =========================================================
# CHAT
# =========================================================

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    if not data:
        return jsonify({
            "error": "No JSON data received."
        }), 400

    message = data.get("message", "").strip()
    mode = data.get("mode", "chat")
    history = data.get("history", [])

    if not message:
        return jsonify({
            "error": "Message cannot be empty."
        }), 400


    if mode == "chat":

        system_instruction = """
You are rast.ai, a helpful AI assistant.

Be clear, friendly, intelligent and useful.

You can help with:
- brainstorming
- writing
- coding
- marketing
- business ideas
- studying
- planning
- research
- creative work

When answering complex questions, organize your
response using headings, bullets and steps.

Never pretend to be human.

Do not claim to have performed an action that you
did not actually perform.
"""

    else:

        system_instruction = """
You are rast.ai Design Studio.

You are an expert:
- creative director
- brand strategist
- graphic designer
- UI/UX designer
- marketing designer
- advertising creative strategist

Help users create:
- Instagram creatives
- advertisements
- brand identities
- logos
- UI concepts
- landing pages
- campaigns
- social media concepts
- visual directions
- design systems
- creative briefs

When useful, structure responses using:

CONCEPT
VISUAL DIRECTION
COLOR PALETTE
TYPOGRAPHY
LAYOUT
COPY
DESIGN DETAILS
NEXT STEPS

Be creative, specific and practical.

Do not claim that an actual image has been generated
unless an image-generation system has actually generated it.
"""


    contents = []

    for item in history:

        role = item.get("role", "user")
        text = item.get("text", "")

        if not text:
            continue

        if role not in ["user", "model"]:
            role = "user"

        contents.append({
            "role": role,
            "parts": [
                {
                    "text": text
                }
            ]
        })


    contents.append({
        "role": "user",
        "parts": [
            {
                "text": message
            }
        ]
    })


    try:

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                max_output_tokens=1500
            )
        )

        reply = response.text

        return jsonify({
            "reply": reply
        })


    except Exception as error:

        print("GEMINI ERROR:", error)

        return jsonify({
            "error": "The AI could not generate a response."
        }), 500


# =========================================================
# IMAGE GENERATION
# =========================================================

@app.route("/generate-image", methods=["POST"])
def generate_image():

    data = request.get_json()

    if not data:
        return jsonify({
            "error": "No JSON data received."
        }), 400


    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({
            "error": "Image prompt cannot be empty."
        }), 400


    try:

        print("IMAGE REQUEST:", prompt)


        image = image_client.text_to_image(
            prompt=prompt,
            model=IMAGE_MODEL
        )


        # Convert generated PIL image into PNG bytes
        buffer = BytesIO()

        image.save(
            buffer,
            format="PNG"
        )

        buffer.seek(0)


        # Convert PNG into base64
        image_base64 = base64.b64encode(
            buffer.read()
        ).decode("utf-8")


        return jsonify({
            "success": True,
            "image": image_base64,
            "format": "png"
        })


    except Exception as error:

        print("IMAGE GENERATION ERROR:", error)

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )

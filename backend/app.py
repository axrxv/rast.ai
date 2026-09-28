import os
import base64

from flask import Flask, request, jsonify
from flask_cors import CORS

from google import genai
from google.genai import types


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)
CORS(app)


# =========================================================
# GEMINI SETUP
# =========================================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=API_KEY)


# =========================================================
# MODELS
# =========================================================

CHAT_MODEL = "gemini-3.5-flash-lite"

IMAGE_MODEL = "gemini-3.1-flash-image"


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

    mode = data.get(
        "mode",
        "chat"
    )

    history = data.get(
        "history",
        []
    )

    if not message:
        return jsonify({
            "error": "Message cannot be empty."
        }), 400


    # -----------------------------------------------------
    # SYSTEM INSTRUCTION
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # BUILD CONVERSATION
    # -----------------------------------------------------

    contents = []

    for item in history:

        role = item.get(
            "role",
            "user"
        )

        text = item.get(
            "text",
            ""
        )

        if not text:
            continue

        if role not in [
            "user",
            "model"
        ]:
            role = "user"

        contents.append(
            types.Content(
                role=role,
                parts=[
                    types.Part(
                        text=text
                    )
                ]
            )
        )


    # -----------------------------------------------------
    # CURRENT USER MESSAGE
    # -----------------------------------------------------

    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part(
                    text=message
                )
            ]
        )
    )


    # -----------------------------------------------------
    # GENERATE RESPONSE
    # -----------------------------------------------------

    try:

        response = client.models.generate_content(

            model=CHAT_MODEL,

            contents=contents,

            config=types.GenerateContentConfig(

                system_instruction=
                    system_instruction,

                temperature=0.7,

                max_output_tokens=1500
            )
        )

        reply = response.text

        return jsonify({
            "reply": reply
        })


    except Exception as error:

        print(
            "GEMINI CHAT ERROR:",
            error
        )

        return jsonify({
            "error":
                "The AI could not generate a response."
        }), 500


# =========================================================
# IMAGE GENERATION
# =========================================================

@app.route(
    "/generate-image",
    methods=["POST"]
)
def generate_image():

    data = request.get_json()

    if not data:

        return jsonify({
            "success": False,
            "error":
                "No JSON data received."
        }), 400


    prompt = data.get(
        "prompt",
        ""
    ).strip()


    if not prompt:

        return jsonify({
            "success": False,
            "error":
                "Image prompt cannot be empty."
        }), 400


    print(
        "IMAGE REQUEST:",
        prompt
    )


    try:

        # -------------------------------------------------
        # GEMINI IMAGE GENERATION
        # -------------------------------------------------

        response = client.models.generate_content(

            model=IMAGE_MODEL,

            contents=[
                prompt
            ],

            config=types.GenerateContentConfig(

                response_modalities=[
                    "IMAGE"
                ],

                response_format={
                    "image": {
                        "aspect_ratio": "16:9",
                        "image_size": "1K"
                    }
                }
            )
        )


        # -------------------------------------------------
        # FIND GENERATED IMAGE
        # -------------------------------------------------

        image_data = None

        for part in response.parts:

            if part.inline_data is not None:

                image_data = (
                    part.inline_data.data
                )

                break


        # -------------------------------------------------
        # CHECK RESULT
        # -------------------------------------------------

        if not image_data:

            raise RuntimeError(
                "Gemini did not return an image."
            )


        # -------------------------------------------------
        # RETURN IMAGE TO WEBSITE
        # -------------------------------------------------

        return jsonify({

            "success": True,

            "image": image_data,

            "format": "png"

        })


    except Exception as error:

        print(
            "IMAGE GENERATION ERROR:",
            error
        )

        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# =========================================================
# RUN SERVER
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

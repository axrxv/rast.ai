import os

from flask import Flask, request, jsonify
from flask_cors import CORS

from google import genai
from google.genai import types


# =========================================================
# APP
# =========================================================

app = Flask(__name__)

CORS(app)


# =========================================================
# GEMINI
# =========================================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")


client = genai.Client(
    api_key=API_KEY
)


MODEL = "gemini-3.5-flash-lite"


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


    message = data.get(
        "message",
        ""
    ).strip()


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


    # =====================================================
    # CHAT MODE
    # =====================================================

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


    # =====================================================
    # DESIGN MODE
    # =====================================================

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


    # =====================================================
    # CONVERSATION
    # =====================================================

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


        # Gemini expects user/model roles.
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


    # Current message

    contents.append({

        "role": "user",

        "parts": [
            {
                "text": message
            }
        ]

    })


    # =====================================================
    # GEMINI REQUEST
    # =====================================================

    try:

        response = client.models.generate_content(

            model=MODEL,

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
            "GEMINI ERROR:",
            error
        )


        return jsonify({

            "error":
                "The AI could not generate a response."

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

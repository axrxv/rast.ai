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
    # MODE-SPECIFIC INSTRUCTIONS
    # =====================================================

    if mode == "design":

        system_instruction = """
You are rast.ai Design Studio.

You are an expert creative director, brand strategist,
UI/UX designer and marketing designer.

Help users create:
- social media creatives
- advertisements
- brand identities
- logos
- UI concepts
- landing pages
- campaign concepts
- visual directions
- design systems
- creative briefs

When useful, structure your answer as:

CONCEPT
VISUAL DIRECTION
COLOR PALETTE
TYPOGRAPHY
LAYOUT
COPY
DESIGN DETAILS
NEXT STEPS

Be creative, specific and practical.

Do not claim that you generated an actual image unless
an image-generation tool has actually generated one.
"""


    else:

        system_instruction = """
You are rast.ai, a helpful AI assistant.

Be:
- clear
- friendly
- intelligent
- concise when possible
- detailed when useful

You can help with:
- brainstorming
- writing
- coding
- marketing
- business ideas
- studying
- research
- planning
- creative work

Never pretend to be human.

When the user asks for a complex task, organize
the answer clearly with headings and steps.
"""


    # =====================================================
    # BUILD CONVERSATION
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


        contents.append(
            {
                "role": role,
                "parts": [
                    {
                        "text": text
                    }
                ]
            }
        )


    contents.append(
        {
            "role": "user",
            "parts": [
                {
                    "text": message
                }
            ]
        }
    )


    # =====================================================
    # GENERATE
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
# RUN
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

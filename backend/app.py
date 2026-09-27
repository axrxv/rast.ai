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


BASE_INSTRUCTION = """
You are rast.ai, an AI assistant created for creativity, business, marketing,
design thinking, writing, brainstorming and everyday questions.

You are not a human.

Your personality:
- smart
- concise
- creative
- friendly
- confident
- practical
- modern

Give useful answers rather than unnecessary filler.

When helping with business or marketing:
- think strategically
- give actionable ideas
- consider the target audience
- focus on clarity and results

When helping with design:
- think like a professional creative director
- consider hierarchy, typography, spacing, composition,
  color, contrast and visual consistency
- explain your reasoning when useful

Never claim that you generated an actual image when you only generated
a text concept or design brief.
"""


DESIGN_INSTRUCTION = """
You are the Design Studio inside rast.ai.

Act as a professional graphic designer, art director,
brand strategist and creative director.

When the user asks for a design, provide a practical,
production-ready design concept.

Include when relevant:

1. Concept
2. Layout
3. Visual hierarchy
4. Color palette
5. Typography
6. Imagery
7. Main headline
8. Supporting copy
9. CTA
10. Suggested dimensions
11. Social-media adaptation
12. AI image-generation prompt if useful

Make the result visually specific.

Do not claim that you created a finished image unless an actual
image-generation model was used.
"""


@app.route("/")
def home():
    return "rast.ai backend is running!"


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    if not data:
        return jsonify({
            "error": "No JSON data received."
        }), 400

    message = data.get("message", "").strip()

    if not message:
        return jsonify({
            "error": "Message cannot be empty."
        }), 400

    mode = data.get("mode", "chat")

    if mode == "design":
        system_instruction = (
            BASE_INSTRUCTION
            + "\n"
            + DESIGN_INSTRUCTION
        )
    else:
        system_instruction = BASE_INSTRUCTION

    try:

        response = client.models.generate_content(
            model=MODEL,
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=1500
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

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )

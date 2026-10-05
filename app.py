import os
import base64
import mimetypes
from typing import List

from flask import Flask, render_template, request
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv(override=True)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class Recipe(BaseModel):
    name: str = Field(description="Name of the recipe")
    description: str = Field(description="Short description of the recipe")
    prep_time: str = Field(description="Estimated preparation time")


class Response(BaseModel):
    ingredients: List[str] = Field(description="List of main ingredients")
    recipes: List[Recipe] = Field(description="List of exactly 3 recipe suggestions")


model = init_chat_model(
    "gemini-3.8-flash",
    model_provider="google_genai"
)

structured_model = model.with_structured_output(Response)

system_prompt = """
You are a helpful chef.

Look at the uploaded image and identify the main ingredients.

Then suggest exactly 3 recipes based on those ingredients.

Return:
- ingredients as a simple list
- recipes as a list of 3 items
- each recipe must contain:
  - name
  - description
  - prep_time

Keep recipe names clear and short.
Keep descriptions short and clean.
"""


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        image = request.files.get("image")

        if image and image.filename:
            image_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(image_path)

            mime_type, _ = mimetypes.guess_type(image_path)

            with open(image_path, "rb") as file:
                raw_binary = file.read()
                base64_bytes = base64.b64encode(raw_binary)
                image_base64 = base64_bytes.decode("utf-8")

            message = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "base64": image_base64,
                            "mime_type": mime_type
                        }
                    ]
                }
            ]

            result = structured_model.invoke(message)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
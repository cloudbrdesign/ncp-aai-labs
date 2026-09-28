"""A customer sends a photo of the shipping label. A vision model reads it; our code looks up the order.

    ollama pull qwen3-vl:8b                      # once (or another vision model, see README)
    python m02/photo_question.py                 # uses m02/data/shipping_label.jpg
    python m02/photo_question.py path/to/photo.jpg

The image goes in the same chat message as the question, as an image_url part with a
base64 data URL, through Ollama's OpenAI-compatible API. Set VLM_BASE_URL, VLM_MODEL
and VLM_API_KEY to use another OpenAI-compatible vision endpoint.
"""
import base64
import json
import os
import pathlib
import re
import sys

from openai import OpenAI

HERE = pathlib.Path(__file__).resolve().parent
ORDERS = json.loads((HERE.parent / "m01" / "data" / "orders.json").read_text())
QUESTION = ("A customer sent this photo of the shipping label on a damaged parcel. "
            "What is the order number on the label, and what item does it say is inside? "
            "Answer in one line: ORDER=<order number>; ITEM=<item>")


def ask_vision_model(image: pathlib.Path) -> str:
    client = OpenAI(base_url=os.environ.get("VLM_BASE_URL", "http://localhost:11434/v1"),
                    api_key=os.environ.get("VLM_API_KEY", "ollama"))  # Ollama ignores the key
    mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
    data_url = f"data:{mime};base64,{base64.b64encode(image.read_bytes()).decode()}"
    reply = client.chat.completions.create(
        model=os.environ.get("VLM_MODEL", "qwen3-vl:8b"),
        messages=[{"role": "user", "content": [
            {"type": "text", "text": QUESTION},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
        temperature=0,
    )
    text = reply.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # drop any visible reasoning


def main():
    image = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data" / "shipping_label.jpg"
    print(f"Photo: {image.name}\nQuestion: {QUESTION}\n")
    answer = ask_vision_model(image)
    print(f"Vision model: {answer}")
    found = re.findall(r"\b[A-Z]\d{4}\b", answer.upper())
    if not found:
        print("\nNo order number found in the answer. Try again, or try a larger vision model.")
        sys.exit(1)
    order_id = found[0]
    order = ORDERS.get(order_id)
    print(f"\nOrder lookup ({order_id}): {json.dumps(order) if order else 'no such order'}")


if __name__ == "__main__":
    main()

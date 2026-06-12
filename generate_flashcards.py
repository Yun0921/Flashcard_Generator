"""
Utilities for reading lecture notes and generating flashcards with Google Gemini.

This file is used by app.py.
It does not handle the web page, saving JSON files, or displaying flashcards.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader


SYSTEM_PROMPT = """You are a study assistant that creates high-quality flashcards from lecture notes.

Rules:
- Extract the most important concepts, definitions, and facts.
- Each card must test ONE specific idea.
- Questions should be clear and concise.
- Answers should be short and specific.
- Write each flashcard in the same language as the lecture notes.
- Avoid vague questions unless the notes define the topic clearly.
- Do not invent facts that are not supported by the notes.

Return ONLY valid JSON in this exact shape:
{
  "flashcards": [
    {"front": "Question here?", "back": "Answer here."}
  ]
}
"""


FALLBACK_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]


def load_pdf(path: Path, max_pages: int | None = None) -> tuple[str, int, int]:
    reader = PdfReader(str(path))
    total_pages = len(reader.pages)

    if total_pages == 0:
        raise ValueError(f"PDF has no pages: {path}")

    pages_to_read = min(total_pages, max_pages) if max_pages else total_pages
    parts: list[str] = []

    for index in range(pages_to_read):
        page_text = reader.pages[index].extract_text() or ""
        page_text = page_text.strip()

        if page_text:
            parts.append(page_text)

    text = "\n\n".join(parts).strip()

    if not text:
        raise ValueError(
            f"Could not extract text from PDF: {path}. "
            "It may be a scanned or image-only PDF."
        )

    return text, pages_to_read, total_pages


def load_notes(path: Path, max_pages: int | None = None) -> tuple[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Notes file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text, pages_read, total_pages = load_pdf(path, max_pages=max_pages)

        if max_pages and pages_read < total_pages:
            detail = f"{pages_read}/{total_pages} pages"
        else:
            detail = f"{total_pages} pages"

        return text, detail

    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8").strip()

        if not text:
            raise ValueError(f"Notes file is empty: {path}")

        return text, "text file"

    raise ValueError(
        f"Unsupported file type '{suffix}'. Please use .txt, .md, or .pdf."
    )


def build_user_prompt(notes: str, max_cards: int) -> str:
    return (
        f"Create up to {max_cards} flashcards from these lecture notes.\n\n"
        f"--- NOTES START ---\n"
        f"{notes}\n"
        f"--- NOTES END ---"
    )


def extract_json_object(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if not match:
            raise ValueError("Model response did not contain valid JSON.")

        return json.loads(match.group(0))


def get_api_key() -> str:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Please add it to your .env file."
        )

    if api_key.startswith("your-") or "貼在這裡" in api_key:
        raise EnvironmentError("請在 .env 填入真正的 GEMINI_API_KEY。")

    return api_key


def normalize_model_name(model: str) -> str:
    cleaned = model.strip().lower().replace(" ", "-")

    if cleaned.startswith("gemini-"):
        return cleaned

    raise ValueError(
        f"Invalid Gemini model name: {model!r}. "
        "Please choose a model from the dropdown list."
    )


def is_quota_error(error: Exception) -> bool:
    message = str(error).upper()
    return "429" in message or "RESOURCE_EXHAUSTED" in message


def is_model_unavailable(error: Exception) -> bool:
    message = str(error).upper()
    return "404" in message or "NOT_FOUND" in message


def generate_with_model(
    client: genai.Client,
    model: str,
    notes: str,
    max_cards: int,
) -> list[dict[str, str]]:
    response = client.models.generate_content(
        model=model,
        contents=build_user_prompt(notes, max_cards),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            response_mime_type="application/json",
        ),
    )

    content = response.text or ""
    payload = extract_json_object(content)
    raw_cards = payload.get("flashcards", [])

    cards: list[dict[str, str]] = []

    for card in raw_cards:
        front = str(card.get("front", "")).strip()
        back = str(card.get("back", "")).strip()

        if front and back:
            cards.append({
                "front": front,
                "back": back
            })

    if not cards:
        raise ValueError("The model returned no usable flashcards.")

    return cards[:max_cards]


def call_llm(
    notes: str,
    max_cards: int,
    model: str = "gemini-2.5-flash-lite",
) -> list[dict[str, str]]:
    client = genai.Client(api_key=get_api_key())

    selected_model = normalize_model_name(model)
    models_to_try = [selected_model]

    for fallback_model in FALLBACK_MODELS:
        if fallback_model != selected_model:
            models_to_try.append(fallback_model)

    last_error: Exception | None = None

    for attempt_model in models_to_try:
        try:
            print(f"Trying model: {attempt_model}")
            return generate_with_model(
                client=client,
                model=attempt_model,
                notes=notes,
                max_cards=max_cards,
            )

        except Exception as error:
            if is_quota_error(error):
                print(f"Quota exceeded for {attempt_model}. Trying next model...")
                last_error = error
                continue

            if is_model_unavailable(error):
                print(f"Model {attempt_model} is unavailable. Trying next model...")
                last_error = error
                continue

            raise

    raise EnvironmentError(
        "Gemini API quota is exhausted or the selected models are unavailable.\n"
        "Please try again later or use another API key.\n"
        f"Last error: {last_error}"
    )
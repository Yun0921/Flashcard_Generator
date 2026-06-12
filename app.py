import json
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from generate_flashcards import load_notes, call_llm


app = Flask(__name__)

UPLOAD_FOLDER = Path("uploads")
OUTPUT_FOLDER = Path("outputs")

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)


AVAILABLE_MODELS = [
    ("gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite (Fastest)"),
    ("gemini-2.5-flash", "Gemini 2.5 Flash (Balanced)"),
    ("gemini-2.5-pro", "Gemini 2.5 Pro (Best Quality)"),
    ("gemini-2.0-flash", "Gemini 2.0 Flash"),
]


def build_combined_notes(files):
    all_notes = []

    for file in files:
        if file.filename == "":
            continue

        filename = secure_filename(file.filename)
        file_path = UPLOAD_FOLDER / filename
        file.save(file_path)

        text, detail = load_notes(file_path)

        all_notes.append(
            f"\n\n===== SOURCE: {filename} ({detail}) =====\n\n{text}"
        )

    if not all_notes:
        raise ValueError("No valid lecture files uploaded.")

    return "\n".join(all_notes)


def validate_flashcards(cards):
    if not isinstance(cards, list):
        raise ValueError("Flashcard file must contain a list.")

    cleaned = []

    for card in cards:
        if not isinstance(card, dict):
            continue

        front = str(card.get("front", "")).strip()
        back = str(card.get("back", "")).strip()
        starred = bool(card.get("starred", False))

        if front and back:
            cleaned.append({
                "front": front,
                "back": back,
                "starred": starred
            })

    if not cleaned:
        raise ValueError("No valid flashcards found.")

    return cleaned


def safe_output_path(filename):
    safe_name = secure_filename(filename)

    if not safe_name:
        raise ValueError("Invalid filename.")

    if not safe_name.endswith(".json"):
        safe_name += ".json"

    return OUTPUT_FOLDER / safe_name


@app.route("/", methods=["GET", "POST"])
def index():
    cards = None
    error = None
    saved_file = None
    original_filename = None

    if request.method == "POST":
        action = request.form.get("action")

        try:
            if action == "generate":
                files = request.files.getlist("notes")
                max_cards = int(request.form.get("max_cards", 10))
                difficulty = request.form.get("difficulty", "medium")
                model = request.form.get("model", "gemini-2.5-flash-lite")

                notes = build_combined_notes(files)

                notes_with_instruction = f"""
Difficulty level: {difficulty}

Please generate flashcards according to this difficulty level:

- easy:
  Focus on basic definitions, simple facts, and beginner-friendly questions.

- medium:
  Include important concepts, comparisons, examples, and common exam points.

- hard:
  Include deeper reasoning, cause-and-effect questions, conceptual differences,
  and exam-style questions that require understanding instead of memorization.

Lecture notes:
{notes}
"""

                cards = call_llm(
                    notes_with_instruction,
                    max_cards=max_cards,
                    model=model
                )

                cards = validate_flashcards(cards)

                # 這是新產生的 deck，所以沒有原始檔名
                original_filename = None

            elif action == "upload_flashcards":
                file = request.files.get("flashcard_file")

                if file is None or file.filename == "":
                    raise ValueError("Please upload a flashcard JSON file.")

                uploaded_cards = json.load(file)
                cards = validate_flashcards(uploaded_cards)

                # 記住使用者上傳的原始檔名
                original_filename = secure_filename(file.filename)

            elif action == "save_flashcards":
                cards_json = request.form.get("cards_json")
                original_filename = request.form.get("original_filename") or None

                if not cards_json:
                    raise ValueError("No flashcards to save.")

                cards = validate_flashcards(json.loads(cards_json))

                if original_filename:
                    # 如果是從 Upload Saved Flashcards 匯入的，就覆蓋原本檔名
                    output_path = safe_output_path(original_filename)
                else:
                    # 如果是從 Upload lecture notes 新產生的，就另存新檔
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = OUTPUT_FOLDER / f"flashcards_{timestamp}.json"

                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(cards, f, ensure_ascii=False, indent=2)

                saved_file = str(output_path)

                # 儲存後繼續記住這個檔名
                original_filename = output_path.name

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        cards=cards,
        error=error,
        saved_file=saved_file,
        original_filename=original_filename,
        models=AVAILABLE_MODELS
    )


if __name__ == "__main__":
    app.run(debug=True)
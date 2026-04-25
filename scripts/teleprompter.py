"""
Step through scripts/DEMO_RECORDING_SCRIPT.md while recording (press Enter between sections).

Optional text-to-speech (Windows-friendly):
  pip install pyttsx3
  python scripts/teleprompter.py --speak

Run from project root:
  python scripts/teleprompter.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent / "DEMO_RECORDING_SCRIPT.md"


def _sections(md: str) -> list[str]:
    """Split on horizontal rules (same structure as DEMO_RECORDING_SCRIPT.md)."""
    parts = [p.strip() for p in md.strip().split("\n---\n")]
    return [p for p in parts if p]


def _speak(text: str) -> None:
    try:
        import pyttsx3  # type: ignore
    except ImportError:
        print("(Install pyttsx3 for TTS: pip install pyttsx3)\n")
        return
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)
    # Strip markdown emphasis and stage directions for speech
    clean = re.sub(r"\*[^*]+\*", "", text)
    clean = re.sub(r"#+\s*", "", clean)
    clean = re.sub(r"\n+", " ", clean).strip()
    if clean:
        engine.say(clean)
        engine.runAndWait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Teleprompter for demo recording.")
    parser.add_argument(
        "--speak",
        action="store_true",
        help="Speak each section with pyttsx3 (pip install pyttsx3).",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=SCRIPT_PATH,
        help="Markdown file to read (default: scripts/DEMO_RECORDING_SCRIPT.md).",
    )
    args = parser.parse_args()

    path: Path = args.file
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    sections = _sections(text)
    print(f"Loaded {len(sections)} section(s) from {path.name}\n")
    print("Tip: Resize this terminal beside your recorder; press Enter after each section.\n")
    print("=" * 60)

    for i, block in enumerate(sections, start=1):
        print(f"\n--- Section {i}/{len(sections)} ---\n")
        print(block)
        print("\n" + "=" * 60)
        if args.speak:
            _speak(block)
        if i < len(sections):
            input("\n[Enter] for next section… ")

    print("\nDone. Good luck with the recording.\n")


if __name__ == "__main__":
    main()

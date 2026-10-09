"""Day 6: render an animated walkthrough GIF (docs/demo.gif) of the app.

Pure PIL -- no browser or display needed. Re-run to regenerate:

    python scripts/make_demo_gif.py

Output: docs/demo.gif (960x600, loops forever).
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 960, 600
BG = (14, 17, 23)
PANEL = (28, 34, 42)
BORDER = (48, 54, 61)
TEXT = (230, 237, 243)
MUTED = (139, 148, 158)
ACCENT = (88, 166, 255)
USER_BUBBLE = (31, 80, 160)
ASSIST_BUBBLE = (28, 34, 42)

OUT = Path(__file__).resolve().parent.parent / "docs" / "demo.gif"


def load_font(bold: bool = False, size: int = 28) -> ImageFont.FreeTypeFont:
    """Prefer DejaVu; fall back to PIL's bitmap font."""
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for candidate in (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts") / name,
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


TITLE = load_font(bold=True, size=44)
H2 = load_font(bold=True, size=30)
BODY = load_font(size=24)
SMALL = load_font(size=20)
TINY = load_font(size=17)


def new_frame() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def wrap(draw: ImageDraw.ImageDraw, text: str,
         font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    """Greedy word wrap to a pixel width."""
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def header(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle([0, 0, W, 64], fill=PANEL)
    draw.text((24, 14), "Learn with Jashwanth", font=H2, fill=TEXT)
    draw.text((24, 46), "Ask anything about the newsletter. "
                        "Answers cite their sources.",
              font=TINY, fill=MUTED)
    draw.line([0, 64, W, 64], fill=BORDER)


def chat_input(draw: ImageDraw.ImageDraw, text: str = "") -> None:
    draw.rectangle([24, H - 64, W - 24, H - 20], outline=BORDER,
                   width=2, fill=PANEL)
    draw.text((40, H - 56), text or "Ask about the newsletter...",
              font=BODY, fill=TEXT if text else MUTED)


def bubble(draw: ImageDraw.ImageDraw, text: str, y: int,
           user: bool = False, font: ImageFont.FreeTypeFont = BODY
           ) -> int:
    """Draw a chat bubble; return the y below it."""
    max_w = W - 220
    lines = wrap(draw, text, font, max_w)
    line_h = font.size + 10
    bw = max(draw.textlength(line, font=font) for line in lines) + 36
    bh = len(lines) * line_h + 24
    x0 = W - 24 - bw if user else 24
    color = USER_BUBBLE if user else ASSIST_BUBBLE
    draw.rounded_rectangle([x0, y, x0 + bw, y + bh], radius=14,
                           fill=color)
    for i, line in enumerate(lines):
        draw.text((x0 + 18, y + 12 + i * line_h), line, font=font,
                  fill=TEXT)
    return y + bh + 16


def frame_title() -> Image.Image:
    img, d = new_frame()
    d.text((W // 2, 180), "Learn with Jashwanth", font=TITLE,
           fill=TEXT, anchor="mm")
    d.text((W // 2, 240), "RAG chatbot over the newsletter",
           font=H2, fill=ACCENT, anchor="mm")
    d.text((W // 2, 300),
           "Ask anything. Every answer cites its source.",
           font=BODY, fill=MUTED, anchor="mm")
    d.rounded_rectangle([W // 2 - 130, 360, W // 2 + 130, 412],
                        radius=12, fill=ACCENT)
    d.text((W // 2, 386), "Watch the demo", font=H2, fill=BG,
           anchor="mm")
    d.text((W // 2, 500), "Week 1 portfolio project  -  Python, "
                          "ChromaDB, Streamlit",
           font=SMALL, fill=MUTED, anchor="mm")
    return img


def frame_suggest() -> Image.Image:
    img, d = new_frame()
    header(d)
    y = 100
    d.text((24, y), "Try one of these:", font=H2, fill=TEXT)
    y += 52
    questions = [
        "How do I clean messy Excel data with pandas?",
        "Do I need machine learning for my first data job?",
        "Why did you start the Learn with Jashwanth newsletter?",
        "What should I learn first as a data analyst?",
    ]
    for q in questions:
        d.rounded_rectangle([24, y, W - 24, y + 52], radius=10,
                            outline=BORDER, width=2)
        d.text((44, y + 12), q, font=BODY, fill=TEXT)
        y += 64
    chat_input(d)
    return img


def frame_question() -> Image.Image:
    img, d = new_frame()
    header(d)
    y = bubble(d, "Why did you start the Learn with Jashwanth "
                  "newsletter?", 100, user=True)
    chat_input(d)
    return img


def frame_searching() -> Image.Image:
    img, d = new_frame()
    header(d)
    y = bubble(d, "Why did you start the Learn with Jashwanth "
                  "newsletter?", 100, user=True)
    bubble(d, "Searching the newsletter...", y, font=SMALL)
    d.text((W // 2, H - 120), "retrieving top-5 chunks -> relevance gate",
           font=TINY, fill=MUTED, anchor="mm")
    chat_input(d)
    return img


ANSWER = ("I started the newsletter because I kept putting off sharing "
          "what I was learning [1]. Writing in public forces me to learn "
          "deeper, and the goal is to help others on the same data/AI "
          "path learn faster [1].")


def frame_answer() -> Image.Image:
    img, d = new_frame()
    header(d)
    y = bubble(d, "Why did you start the Learn with Jashwanth "
                  "newsletter?", 100, user=True)
    y = bubble(d, ANSWER, y)
    d.rounded_rectangle([24, y, W - 24, y + 44], radius=8,
                        outline=BORDER, width=2)
    d.text((44, y + 10), "Sources (1)", font=SMALL, fill=MUTED)
    d.text((W - 44, y + 10), "backend: extractive  -  0.4s",
           font=TINY, fill=MUTED, anchor="rm")
    chat_input(d)
    return img


def frame_sources() -> Image.Image:
    img, d = new_frame()
    header(d)
    y = bubble(d, "Why did you start the Learn with Jashwanth "
                  "newsletter?", 100, user=True)
    y = bubble(d, ANSWER, y)
    d.rounded_rectangle([24, y, W - 24, y + 100], radius=8,
                        outline=BORDER, width=2, fill=PANEL)
    d.text((44, y + 10), "Sources", font=SMALL, fill=MUTED)
    d.text((44, y + 40),
           "[1] I finally started the newsletter I kept putting off",
           font=SMALL, fill=ACCENT)
    d.text((44, y + 66), "learnwithjashwanth.substack.com",
           font=TINY, fill=MUTED)
    chat_input(d)
    return img


def frame_end() -> Image.Image:
    img, d = new_frame()
    d.text((W // 2, 130), "What is under the hood", font=TITLE,
           fill=TEXT, anchor="mm")
    bullets = [
        "Grounded answers with [1]-style citations",
        "Honest abstention via a relevance gate",
        "Cached answers - repeat questions are instant",
        "Two backends: extractive + HuggingFace, no API key needed",
        "10-question eval harness with before/after results",
    ]
    y = 210
    for b in bullets:
        d.text((W // 2 - 330, y), "- " + b, font=BODY, fill=TEXT)
        y += 48
    d.text((W // 2, 500),
           "github.com/dasarijashwanth/learnwithjashwanth-rag",
           font=SMALL, fill=ACCENT, anchor="mm")
    return img


def main() -> None:
    frames = [
        (frame_title(), 2200),
        (frame_suggest(), 1800),
        (frame_question(), 1400),
        (frame_searching(), 1400),
        (frame_answer(), 2200),
        (frame_sources(), 2200),
        (frame_end(), 2600),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    images, durations = zip(*frames)
    images[0].save(
        OUT, save_all=True, append_images=images[1:],
        duration=list(durations), loop=0,
    )
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

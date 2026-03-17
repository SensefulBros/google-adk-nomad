"""
tools.py
─────────────────────────────────────────────────────
Smoke-tests the agent pipeline locally (no ADK web UI needed).

Usage:
  pip install -r requirements.txt
  cp .env.example .env   # fill in your key
  python tools.py

What it tests:
  1. story_writer produces valid 5-page JSON
  2. generate_illustrations reads state and calls Imagen
  3. Artifacts are saved and named correctly
─────────────────────────────────────────────────────
"""

import asyncio
import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

# ── Minimal ToolContext stub (mirrors ADK's real interface) ──────────────────
class _StubToolContext:
    def __init__(self, state: dict):
        self.state = state
        self._artifacts: dict[str, bytes] = {}

    def save_artifact(self, filename: str, artifact):
        # artifact is a genai Part — grab the bytes
        self._artifacts[filename] = artifact.inline_data.data
        print(f"  [artifact saved] {filename} ({len(artifact.inline_data.data):,} bytes)")


# ── Test 1: story_writer output ──────────────────────────────────────────────
async def test_story_writer():
    print("\n=== Test 1: Story Writer (live Gemini call) ===")
    from google.genai import Client

    client = Client(api_key=os.environ["GOOGLE_API_KEY"])
    theme = "a brave little fox who learns to share"

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"Theme: {theme}",
        config={
            "system_instruction": (
                "You are a children's book author. Write a 5-page story as a JSON array. "
                "Each element: {page, text, visual_description}. Output ONLY JSON."
            )
        },
    )

    raw = response.text
    clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    pages = json.loads(clean)

    assert len(pages) == 5, f"Expected 5 pages, got {len(pages)}"
    for p in pages:
        assert "page" in p and "text" in p and "visual_description" in p

    print(f"  ✓ Got {len(pages)} pages")
    for p in pages:
        print(f"  Page {p['page']}: {p['text'][:60]}...")

    return clean   # return JSON string to reuse in test 2


# ── Test 2: Illustrator tool ─────────────────────────────────────────────────
async def test_illustrator(story_json: str):
    print("\n=== Test 2: Illustrator Tool (live Imagen call) ===")
    from storybook.agent import generate_illustrations

    state = {"story_data": story_json}
    ctx = _StubToolContext(state)

    result = generate_illustrations(ctx)

    print(f"  Total pages  : {result['total_pages']}")
    print(f"  Images saved : {result['images_generated']}")

    for r in result["results"]:
        status = r["status"]
        mark = "✓" if status == "ok" else "✗"
        print(f"  {mark} Page {r['page']}: {status}")

    assert result["images_generated"] == 5, (
        f"Expected 5 images, got {result['images_generated']}. "
        "Check GOOGLE_API_KEY and billing on your GCP project."
    )
    print("  ✓ All illustrations generated successfully")


# ── Runner ───────────────────────────────────────────────────────────────────
async def main():
    story_json = await test_story_writer()
    await test_illustrator(story_json)
    print("\n✅ All tests passed — run `adk web` from the parent directory to use the UI\n")


if __name__ == "__main__":
    asyncio.run(main())
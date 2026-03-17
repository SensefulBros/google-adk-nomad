"""
Storybook Multi-Agent System
─────────────────────────────────────────────────────────────────────────────
Architecture:
  root_agent (SequentialAgent)
    ├── story_writer  → writes 5-page story as JSON → state["story_data"]
    └── illustrator   → reads state["story_data"], calls Imagen, saves artifacts

Run:
  cd .. && adk web
  Then open http://localhost:8000 and select "storybook"
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
import re

from google.adk.agents import Agent, SequentialAgent
from google.adk.tools import ToolContext
from google.genai import Client
from google.genai import types as genai_types

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Imagen client (initialised lazily so missing key doesn't crash import) ─
_genai_client: Client | None = None


def _get_client() -> Client:
    global _genai_client
    if _genai_client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set in .env")
        _genai_client = Client(api_key=api_key)
    return _genai_client


# ─── Tool: generate_illustrations ───────────────────────────────────────────

def generate_illustrations(tool_context: ToolContext) -> dict:
    """
    Reads story_data from agent state, generates one image per page via
    Imagen 3, saves each as an ADK artifact, and returns a summary.

    State contract
    ──────────────
    Input  : state["story_data"]  — JSON string produced by story_writer
    Output : state["illustrations"] — list of artifact filenames (set here)
    """
    # ── 1. Pull story data from shared state ────────────────────────────────
    raw = tool_context.state.get("story_data", "")
    if not raw:
        return {"error": "story_data not found in state. Run story_writer first."}

    # story_writer output_key stores the model's raw text response.
    # The model is instructed to output ONLY JSON, but may wrap it in
    # ```json … ``` fences — strip those defensively.
    clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

    try:
        pages: list[dict] = json.loads(clean)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse story_data: %s\nRaw:\n%s", exc, raw)
        return {"error": f"Could not parse story_data JSON: {exc}"}

    client = _get_client()
    artifacts_saved: list[str] = []
    results: list[dict] = []

    # ── 2. Generate one image per page ──────────────────────────────────────
    for page in pages:
        page_num: int = page.get("page", 0)
        text: str = page.get("text", "")
        visual: str = page.get("visual_description", "")

        # Build a child-safe, vivid prompt
        prompt = (
            f"Children's storybook illustration, soft watercolor style, "
            f"warm and whimsical, suitable for ages 4-8. "
            f"Scene: {visual}. "
            f"Do NOT include any text or letters in the image."
        )

        logger.info("Generating image for page %d — prompt: %s", page_num, prompt[:80])

        try:
            response = client.models.generate_images(
                model="imagen-3.0-generate-002",
                prompt=prompt,
                config=genai_types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio="4:3",           # landscape — fits storybook pages
                    safety_filter_level="block_low_and_above",
                    person_generation="allow_adult",  # allow animal characters etc.
                ),
            )

            if not response.generated_images:
                logger.warning("No image returned for page %d", page_num)
                results.append({"page": page_num, "status": "no_image_returned"})
                continue

            image_bytes: bytes = response.generated_images[0].image.image_bytes
            filename = f"page_{page_num:02d}.png"

            # ── 3. Save to ADK artifact store ────────────────────────────────
            tool_context.save_artifact(
                filename=filename,
                artifact=genai_types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png",
                ),
            )

            artifacts_saved.append(filename)
            results.append(
                {
                    "page": page_num,
                    "status": "ok",
                    "artifact": filename,
                    "text": text,
                    "visual": visual,
                }
            )
            logger.info("Saved artifact: %s", filename)

        except Exception as exc:  # noqa: BLE001
            logger.error("Image generation failed for page %d: %s", page_num, exc)
            results.append({"page": page_num, "status": "error", "detail": str(exc)})

    # ── 4. Persist artifact list to state for downstream use ────────────────
    tool_context.state["illustrations"] = artifacts_saved

    return {
        "total_pages": len(pages),
        "images_generated": len(artifacts_saved),
        "results": results,
    }


# ─── Agent 1: Story Writer ───────────────────────────────────────────────────
story_writer = Agent(
    name="story_writer",
    model="gemini-2.0-flash",
    description=(
        "Writes a 5-page children's story as structured JSON when given a theme."
    ),
    instruction="""
You are a warm, imaginative children's book author who writes for ages 4–8.

When the user gives you a theme or topic, write a 5-page story.

OUTPUT FORMAT — respond with ONLY valid JSON (no markdown fences, no extra text):

[
  {
    "page": 1,
    "text": "Once upon a time...",
    "visual_description": "A small white rabbit standing in front of a mushroom house"
  },
  {
    "page": 2,
    "text": "...",
    "visual_description": "..."
  },
  ...
]

Rules:
- text: 1–3 short, simple sentences a child can follow.
- visual_description: concise scene description for an illustrator (no dialog, no text in image).
- Keep all content age-appropriate, positive, and imaginative.
- Output ONLY the JSON array. Nothing else.
""",
    output_key="story_data",   # ← writes model output to state["story_data"]
)

# ─── Agent 2: Illustrator ────────────────────────────────────────────────────
illustrator = Agent(
    name="illustrator",
    model="gemini-2.0-flash",
    description="Reads story_data from state and generates illustrations via Imagen.",
    instruction="""
You are a storybook art director.

Your ONLY job is to call the `generate_illustrations` tool — do not skip it.

After the tool returns, format the output as a readable storybook summary:

For each page in the tool result, display:
  📖 Page N
  Text: <story text>
  🎨 Visual: <visual description>
  🖼️ Image: <artifact filename or error note>

End with a friendly line like "Your storybook is ready! 🎉"
""",
    tools=[generate_illustrations],
)

# ─── Root: SequentialAgent ───────────────────────────────────────────────────
root_agent = SequentialAgent(
    name="storybook_pipeline",
    description=(
        "Give me a theme (e.g. 'a brave little fox') and I'll write and "
        "illustrate a 5-page children's storybook."
    ),
    sub_agents=[story_writer, illustrator],
)
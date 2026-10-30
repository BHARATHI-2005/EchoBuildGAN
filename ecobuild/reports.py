from __future__ import annotations


def build_design_report(
    *,
    description: str,
    preferences: list[str],
    original_filename: str | None,
    generated: bool,
) -> str:
    """Create a factual Markdown report without unverified performance claims."""
    preference_text = (
        "\n".join(f"- {preference}" for preference in preferences)
        if preferences
        else "- No preferences selected"
    )
    image_text = original_filename or "No image uploaded"
    status_text = (
        "Generated image available"
        if generated
        else "No generated image available"
    )
    project_text = description.strip() or "No project description provided."
    return f"""# EcoBuild GAN — Design Concept Report

> This report describes an AI-generated visual concept, not a verified
> architectural plan or engineering recommendation.

## Project description

{project_text}

## Selected sustainability preferences

{preference_text}

## Image and generation status

- Original image: {image_text}
- Generation: {status_text}

## Suggested next steps

- Review the concept with a licensed architect and relevant building professionals.
- Assess structural safety, local code compliance, energy performance, costs,
  materials, and environmental impact using appropriate professional methods.
- Treat selected preferences as descriptive suggestions; they do not condition
  the current Pix2Pix generator.
"""

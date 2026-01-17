import os
import base64
import asyncio
from pathlib import Path
from typing import Tuple, Optional
from google import genai
from google.genai import types
from PIL import Image

# Model for image generation
IMAGE_MODEL = "gemini-2.5-flash-image"

async def generate_image_from_prompt(
    prompt: str,
    output_path: Path,
    api_key: str,
    retry_count: int = 3,
    aspect_ratio: str = "1:1"
) -> Tuple[bool, Optional[str]]:
    """
    Generate image using Gemini Nano Banana Pro image generation model.

    Args:
        prompt: Detailed description of image to generate (write like a creative director)
        output_path: Path where JPEG file should be saved
        api_key: Google API key (GEMINI_API_KEY)
        retry_count: Number of retries on failure
        aspect_ratio: Image aspect ratio (e.g., "1:1", "16:9", "3:4")

    Returns:
        Tuple of (success: bool, base64_data: Optional[str])
    """
    # Validate API key
    if not api_key:
        print("ERROR: API key is missing. Please set GOOGLE_API_KEY in your .env file.")
        return False, None
    
    # Pass API key directly to client like other files do
    client = genai.Client(api_key=api_key)

    for attempt in range(retry_count):
        try:
            # Use generate_content with image model
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=IMAGE_MODEL,  # "gemini-2.5-flash-image"
                    contents=[prompt]
                )
            )

            # Extract image from response parts
            image = None
            for part in response.parts:
                if part.inline_data is not None:
                    image = part.as_image()  # Returns PIL Image
                    break

            if image:
                # Save directly to file (no format parameter needed)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(str(output_path))
                
                # Read back bytes for base64 encoding
                image_bytes = output_path.read_bytes()
                base64_data = base64.b64encode(image_bytes).decode('utf-8')
                
                return True, base64_data
            else:
                print(f"Image generation returned no images (attempt {attempt + 1}/{retry_count})")
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue

        except Exception as e:
            if attempt == retry_count - 1:
                print(f"Failed to generate image after {retry_count} attempts: {e}")
                return False, None
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

    return False, None

def create_image_filename(
    course_name: str,
    question_type: str,
    unit_id: str,
    set_index: int,
    question_index: int
) -> str:
    """Create standardized filename for generated images."""
    return f"{course_name}_{question_type}_u{unit_id}_s{set_index}_q{question_index}.jpeg"

def enhance_prompt_for_image_generation(raw_prompt: str, question_stem: str = "", question_type: str = "MCQ", course_name: str = "", answer_choices: str = "", correct_answer_index: int = -1) -> str:
    """
    Enhance the LLM-generated image prompt with question context for relevance.
    
    Args:
        raw_prompt: The raw prompt from the LLM (after "IMAGE_PROMPT:")
        question_stem: The question text that this image will support
        question_type: "MCQ" or "FRQ"
        course_name: AP course name (e.g., "AP Physics 1")
        answer_choices: The answer choices (for MCQ only)
        correct_answer_index: Index of correct answer (0-3 for MCQ, -1 for FRQ/none)
        
    Returns:
        Enhanced prompt with question context and explicit rendering rules
    """
    # Build context header
    context_header = f"You are creating an image for an {course_name} {question_type} exam.\n\n"
    
    if question_stem:
        context_header += f"QUESTION CONTEXT:\n{question_stem}\n"
    
    if answer_choices and question_type == "MCQ":
        context_header += f"\nANSWER CHOICES:\n{answer_choices}\n"
        
        # Add correct answer indication
        if correct_answer_index >= 0:
            choices_list = answer_choices.split('\n')
            if correct_answer_index < len(choices_list):
                context_header += f"\nCORRECT ANSWER: {choices_list[correct_answer_index]}\n"
    
    context_header += "\n"
    
    # Critical rendering rules
    rendering_rules = (
        "CRITICAL RENDERING RULES:\n"
        "- Create ONLY the visual stimulus described below\n"
        "- DO NOT include the question text in the image\n"
        "- DO NOT include answer choices (A/B/C/D) in the image\n"
        "- DO NOT add question-related titles or captions\n"
        "- The image should be a clean, standalone visual that students reference to answer the question\n"
        "- Include only data labels, axis labels, diagram labels (not question-related text)\n\n"
    )
    
    # Quality requirements
    quality_suffix = (
        "\n\nQUALITY REQUIREMENTS:\n"
        "- Highly detailed and visually clear\n"
        "- Professional AP exam styling\n"
        "- Clean lines and high contrast\n"
        "- Well-composed and suitable for standardized testing\n"
    )
    
    return context_header + rendering_rules + "IMAGE TO CREATE:\n" + raw_prompt + quality_suffix

"""
Phase 1: Content Extraction
Extracts content from single CED PDFs using Gemini 2.5 Pro Vision.

Process:
1. Analyze PDF structure to find section boundaries
2. Extract each section separately (skills, big ideas, units, exam sections, task verbs)
3. Normalize to unified schema
4. Validate and save
"""

import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from jsonschema import validate, ValidationError
import pdf2image
from PIL import Image
import io
import certifi
import ssl

# Fix SSL certificate issue on macOS
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
assert API_KEY, "Missing GEMINI_API_KEY"
client = genai.Client(api_key=API_KEY)

# Constants
COURSES_DIR = "courses"
CONFIG_DIR = "utils/config"
CONTENT_DIR = "utils/content"
SCHEMA_PATH = "utils/schemas/content_schema.json"
MODEL_VISION = "gemini-2.0-flash-exp"  # Vision model for PDF structure analysis
MODEL_EXTRACTION = "gemini-2.5-pro"  # Model for section extraction (can use vision or text)
MAX_IMAGES_PER_REQUEST = 16  # Gemini vision limit per request

# Testing mode: extract only one unit
TEST_MODE = True  # Set to False to extract all units

def load_schema():
    """Load the content schema for validation."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_extraction_config(course_id):
    """Load extraction config for a course."""
    config_path = os.path.join(CONFIG_DIR, f"{course_id}_extraction.json")
    
    if not os.path.exists(config_path):
        # Create default config if it doesn't exist
        default_config = {
            "course_id": course_id,
            "pdf_file": f"courses/{course_id}.pdf",
            "extraction_method": "vision_model",
            "model": MODEL_VISION,
            "section_guide": {
                "skills": {
                    "detection_method": "auto",
                    "hints": ["Skills", "Science Practices", "Course Skills"],
                    "required": True
                },
                "big_ideas": {
                    "detection_method": "auto",
                    "hints": ["Big Ideas", "Themes", "Enduring Understandings"],
                    "required": True
                },
                "units": {
                    "detection_method": "auto",
                    "hints": ["Unit 1", "Unit 2", "Course Units", "Module"],
                    "required": True,
                    "extract_each_unit_separately": True
                },
                "exam_sections": {
                    "detection_method": "auto",
                    "hints": ["Exam Information", "Section I", "Section II", "Multiple Choice", "Free Response"],
                    "required": True
                },
                "task_verbs": {
                    "detection_method": "auto",
                    "hints": ["Task Verbs", "Command Terms", "Action Verbs"],
                    "required": False
                }
            }
        }
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2)
        return default_config
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def pdf_to_images(pdf_path, max_pages=None):
    """Convert PDF pages to images."""
    try:
        images = pdf2image.convert_from_path(pdf_path, dpi=200)
        if max_pages:
            images = images[:max_pages]
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

def analyze_pdf_structure(pdf_path, config):
    """Stage 1: Analyze PDF structure to find section boundaries with multi-pass search."""
    print(f"📊 Analyzing PDF structure: {pdf_path}")
    
    # Get total page count first
    all_images = pdf_to_images(pdf_path)
    total_pages = len(all_images)
    print(f"   📄 Total pages in PDF: {total_pages}")
    
    if not all_images:
        raise RuntimeError(f"Failed to convert PDF to images: {pdf_path}")
    
    # Prepare prompt for structure analysis
    section_guide = config["section_guide"]
    hints_text = "\n".join([
        f"- {section}: {section_info.get('hints', [])}"
        for section, section_info in section_guide.items()
    ])
    
    # Pass 1: Analyze first 30 pages for main sections (skills, big ideas, units)
    print(f"   🔍 Pass 1: Analyzing first 30 pages for main sections...")
    images_pass1 = all_images[:min(30, total_pages)]
    
    prompt_pass1 = f"""Analyze this AP {config['course_id'].replace('ap_', '').replace('_', ' ').title()} CED PDF.

You MUST identify the EXACT page ranges for the following sections:
1. Skills section (with subskills and codes like "1.A", "2.B", etc.) - look for headers like "Skills", "Science Practices", "Course Skills"
2. Big Ideas/Themes section - look for headers like "Big Ideas", "Themes", "Enduring Understandings"
3. Course Units - look for "Unit 1", "Unit 2", etc. and list ALL units with their EXACT page ranges
4. Exam Information (MCQ and FRQ sections) - look for "Section I", "Section II", "Multiple Choice", "Free Response"
5. Task Verbs (if present) - look for "Task Verbs", "Command Terms", "Action Verbs"

Look for these hints in the PDF:
{hints_text}

Return ONLY valid JSON in this exact format:
{{
  "skills": {{"start_page": <number>, "end_page": <number>}},
  "big_ideas": {{"start_page": <number>, "end_page": <number>}},
  "units": [
    {{"name": "Unit 1: <name>", "start_page": <number>, "end_page": <number>}},
    {{"name": "Unit 2: <name>", "start_page": <number>, "end_page": <number>}},
    ...
  ],
  "exam_sections": {{"start_page": <number>, "end_page": <number>}} or null,
  "task_verbs": {{"start_page": <number>, "end_page": <number>}} or null
}}

CRITICAL:
- Use 1-based page numbering (first page is 1)
- Be ACCURATE with page ranges - include the entire section
- Include ALL units you find - do not skip any
- If a section is not found, set it to null
- For units, make sure to include the complete unit name from the PDF
"""
    
    image_parts_pass1 = []
    for img in images_pass1:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        image_parts_pass1.append({
            "mime_type": "image/png",
            "data": img_bytes.read()
        })
    
    try:
        response_pass1 = client.models.generate_content(
            model=MODEL_VISION,
            contents=[{"parts": [{"text": prompt_pass1}] + [{"inline_data": img} for img in image_parts_pass1]}],
            config={"response_mime_type": "application/json"}
        )
        
        if not response_pass1 or not response_pass1.text:
            raise RuntimeError("Vision model returned empty response")
        
        structure = json.loads(response_pass1.text)
        print(f"   ✅ Pass 1 complete")
        print(f"      Skills: pages {structure.get('skills', {}).get('start_page')}-{structure.get('skills', {}).get('end_page') if structure.get('skills') else 'N/A'}")
        print(f"      Big Ideas: pages {structure.get('big_ideas', {}).get('start_page')}-{structure.get('big_ideas', {}).get('end_page') if structure.get('big_ideas') else 'N/A'}")
        print(f"      Units: {len(structure.get('units', []))} units found")
        print(f"      Exam Sections: {'Found' if structure.get('exam_sections') else 'Not found in first 30 pages'}")
        print(f"      Task Verbs: {'Found' if structure.get('task_verbs') else 'Not found in first 30 pages'}")
        
        # Pass 2: Search for exam sections and task verbs if not found (often near the end)
        if not structure.get("exam_sections") or not structure.get("task_verbs"):
            print(f"   🔍 Pass 2: Searching remaining pages for exam sections and task verbs...")
            
            # Search from middle to end of PDF
            start_search_page = min(30, total_pages - 1)
            end_search_page = total_pages
            search_images = all_images[start_search_page:end_search_page]
            
            if search_images:
                # Sample pages from the search range (every 5th page to stay within limits)
                sample_indices = list(range(0, len(search_images), max(1, len(search_images) // 15)))
                sample_images = [search_images[i] for i in sample_indices[:15]]  # Max 15 images
                
                prompt_pass2 = f"""Search these pages (pages {start_search_page + 1} to {end_search_page}) of the AP {config['course_id'].replace('ap_', '').replace('_', ' ').title()} CED PDF.

Find the EXACT page ranges for:
1. Exam Information section - look for "Section I", "Section II", "Multiple Choice", "Free Response", "Exam Information", "AP Exam"
2. Task Verbs section - look VERY carefully for:
   - Headers like "Task Verbs", "Command Terms", "Action Verbs", "Verbs", "Task Verb Definitions"
   - Tables or lists that define verbs like "Calculate", "Justify", "Explain", "Describe", "Compare", "Construct", "Determine", "Identify", "Interpret", "Verify"
   - Sections that explain what students must do when they see certain verbs in FRQ questions
   - Task verbs might be embedded within the exam section or in a separate appendix

IMPORTANT: Task verbs are CRITICAL for FRQ generation. They are often:
- Near the end of the document
- In an appendix section
- Embedded within the exam information section
- Listed in a table format with verb names and descriptions

Look for patterns like:
- "Calculate: Perform mathematical steps..."
- "Justify: Provide evidence..."
- A table with columns for "Verb" and "Description"
- A list of verbs with explanations

Return ONLY valid JSON:
{{
  "exam_sections": {{"start_page": <number>, "end_page": <number>}} or null,
  "task_verbs": {{"start_page": <number>, "end_page": <number>}} or null
}}

CRITICAL: If you see ANY mention of task verbs, verb definitions, or command terms, provide the page range. Do not return null for task_verbs unless you are absolutely certain they are not present.
"""
                
                image_parts_pass2 = []
                for img in sample_images:
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    image_parts_pass2.append({
                        "mime_type": "image/png",
                        "data": img_bytes.read()
                    })
                
                try:
                    response_pass2 = client.models.generate_content(
                        model=MODEL_VISION,
                        contents=[{"parts": [{"text": prompt_pass2}] + [{"inline_data": img} for img in image_parts_pass2]}],
                        config={"response_mime_type": "application/json"}
                    )
                    
                    if response_pass2 and response_pass2.text:
                        structure_pass2 = json.loads(response_pass2.text)
                        
                        # Merge results
                        if structure_pass2.get("exam_sections") and not structure.get("exam_sections"):
                            structure["exam_sections"] = structure_pass2["exam_sections"]
                            print(f"      ✅ Exam Sections found: pages {structure['exam_sections']['start_page']}-{structure['exam_sections']['end_page']}")
                        
                        if structure_pass2.get("task_verbs") and not structure.get("task_verbs"):
                            structure["task_verbs"] = structure_pass2["task_verbs"]
                            print(f"      ✅ Task Verbs found: pages {structure['task_verbs']['start_page']}-{structure['task_verbs']['end_page']}")
                except Exception as e:
                    print(f"      ⚠️  Pass 2 search had issues: {e}")
        
        print(f"✅ PDF structure analysis complete")
        return structure
        
    except Exception as e:
        print(f"❌ Error analyzing PDF structure: {e}")
        raise

def extract_section(pdf_path, section_name, page_range, section_type, schema_template, max_retries=3):
    """Stage 2: Extract a specific section from PDF with chunking and retry logic."""
    if not page_range or page_range.get("start_page") is None:
        print(f"   ⚠️  {section_name} section not found, skipping")
        return None
    
    print(f"📄 Extracting {section_name} (pages {page_range['start_page']}-{page_range['end_page']})")
    
    # Convert relevant pages to images
    start_page = page_range["start_page"] - 1  # Convert to 0-based
    end_page = page_range["end_page"]
    images = pdf_to_images(pdf_path)
    section_images = images[start_page:end_page]
    
    if not section_images:
        print(f"   ⚠️  No images extracted for {section_name}")
        return None
    
    # Chunk images if needed (Gemini vision limit is 16 images per request)
    if len(section_images) > MAX_IMAGES_PER_REQUEST:
        print(f"   📦 Section spans {len(section_images)} pages, chunking into batches of {MAX_IMAGES_PER_REQUEST}")
        chunks = []
        for i in range(0, len(section_images), MAX_IMAGES_PER_REQUEST):
            chunks.append(section_images[i:i + MAX_IMAGES_PER_REQUEST])
        
        # Extract from each chunk and merge
        all_extracted = []
        for chunk_idx, chunk_images in enumerate(chunks):
            chunk_start_page = start_page + (chunk_idx * MAX_IMAGES_PER_REQUEST) + 1
            chunk_end_page = min(start_page + ((chunk_idx + 1) * MAX_IMAGES_PER_REQUEST) + 1, end_page)
            print(f"   📄 Processing chunk {chunk_idx + 1}/{len(chunks)} (pages {chunk_start_page}-{chunk_end_page})")
            
            chunk_result = _extract_section_chunk(
                chunk_images, section_name, section_type, chunk_idx, len(chunks), max_retries
            )
            if chunk_result:
                all_extracted.append(chunk_result)
        
        # Merge results
        return _merge_section_results(all_extracted, section_type)
    else:
        # Single chunk extraction
        return _extract_section_chunk(section_images, section_name, section_type, 0, 1, max_retries)

def _extract_section_chunk(images, section_name, section_type, chunk_idx, total_chunks, max_retries):
    """Extract a single chunk of images."""
    
    # Build extraction prompt based on section type
    chunk_context = ""
    if total_chunks > 1:
        chunk_context = f"\n\nIMPORTANT: This is chunk {chunk_idx + 1} of {total_chunks}. Extract ALL items from these pages. Do not skip any skills, subskills, or other items."
    
    if section_type == "skills":
        prompt = f"""Extract ALL skills from these pages of the AP CED PDF.

You MUST extract:
- ALL skill categories (e.g., "Selecting Statistical Methods", "Data Analysis", "Using Probability and Simulation", "Argumentation")
- ALL subskills with their codes (e.g., "1.A", "1.B", "2.A", "2.B", "3.A", "4.A", etc.)
- Complete skill descriptions
- Complete subskill descriptions

Look for:
- Skill category headers
- Subskill codes in format like "1.A", "1.B", "2.A", etc.
- Descriptions for each skill and subskill

Return ONLY valid JSON matching this schema:
{{
  "skills": [
    {{
      "skill_name": "<category name>",
      "skill_description": "<complete description>",
      "subskills": [
        {{
          "subskill_name": "<code like 1.A>",
          "subskill_description": "<complete description>"
        }}
      ]
    }}
  ]
}}

CRITICAL: Extract EVERY skill category and EVERY subskill. Do not omit any.{chunk_context}"""
    
    elif section_type == "big_ideas":
        prompt = f"""Extract ALL big ideas/themes from these pages of the AP CED PDF.

You MUST extract:
- ALL big idea IDs (e.g., "VAR", "UNC", "DAT" - these are typically 3-letter codes)
- Complete big idea names
- Complete big idea descriptions (these are usually full paragraphs)

Look for:
- Big idea identifiers (usually 3-letter codes)
- Full names of big ideas
- Detailed descriptions explaining each big idea

Return ONLY valid JSON matching this schema:
{{
  "big_ideas": [
    {{
      "id": "<ID like VAR, UNC, DAT>",
      "name": "<Full name>",
      "description": "<Complete description paragraph>"
    }}
  ]
}}

CRITICAL: Extract EVERY big idea. Do not omit any.{chunk_context}"""
    
    elif section_type == "exam_sections":
        prompt = f"""Extract ALL exam section information from these pages of the AP CED PDF.

You MUST extract COMPLETE information for BOTH sections:

**Section I (Multiple-choice):**
- Number of questions
- Timing (e.g., "90 minutes")
- Exam weighting (e.g., "50%")
- ALL descriptions including:
  * Unit weightings (e.g., "Unit 1: Exploring One-Variable Data 15–23%")
  * Skill weightings (e.g., "Skill 1: Selecting Statistical Methods 15–23%")
  * Question format details
  * Assessment expectations
  * Any bullet points or paragraphs describing the section

**Section II (Free-response):**
- Number of questions
- Timing (e.g., "90 minutes")
- Exam weighting (e.g., "50%")
- ALL descriptions including:
  * Part A and Part B information
  * Question focus areas
  * Skill assessments
  * Any bullet points or paragraphs describing the section

IMPORTANT: 
- Extract EVERY sentence, bullet point, and paragraph that describes each section
- Each description should be a separate string in the descriptions array
- Do NOT combine multiple descriptions into one string
- Include ALL text, even if it seems repetitive

Return ONLY valid JSON matching this schema:
{{
  "exam_sections": [
    {{
      "section": "I",
      "question_type": "Multiple-choice questions",
      "Number of Questions": "<number>",
      "exam_weighting": "<percentage>",
      "timing": "<time>",
      "descriptions": [
        "<first description>",
        "<second description>",
        "<third description>",
        "... continue for ALL descriptions ..."
      ]
    }},
    {{
      "section": "II",
      "question_type": "Free-response questions",
      "Number of Questions": "<number>",
      "exam_weighting": "<percentage>",
      "timing": "<time>",
      "descriptions": [
        "<first description>",
        "<second description>",
        "<third description>",
        "... continue for ALL descriptions ..."
      ]
    }}
  ]
}}

CRITICAL: 
- Extract EVERY description as a separate string in the array
- The descriptions array should have MANY items (often 10+ for each section)
- Include ALL bullet points, paragraphs, and details
- Do NOT skip any text describing the exam sections{chunk_context}"""
    
    elif section_type == "task_verbs":
        prompt = f"""Extract ALL task verbs from these pages of the AP CED PDF.

Task verbs are action words used in free-response questions (FRQs) that tell students what they need to do. Examples include: "Calculate", "Justify", "Explain", "Describe", "Compare", "Construct", "Determine", "Estimate", "Give examples", "Identify", "Interpret", "Verify", etc.

You MUST extract:
- ALL task verbs (look for words like: Calculate, Compare, Construct/Complete, Describe, Determine, Estimate, Explain, Give a point estimate, Give examples, Identify/Indicate/Circle, Interpret, Justify, Verify, and any others)
- Complete verb descriptions (what students must do when this verb is used in FRQ questions)

Look for:
- A section titled "Task Verbs", "Command Terms", "Action Verbs", "Verbs", or similar
- Verb names (usually bold, in a table, or listed with descriptions)
- Descriptions explaining what each verb means in the context of the exam
- Tables or lists that pair verbs with their definitions
- Text like "When students see 'Calculate', they must..." or "Calculate: Perform mathematical steps..."

The task verbs section might be:
- In a dedicated section near the end of the document
- Embedded within the exam information section
- In an appendix
- Listed in a table format

Return ONLY valid JSON matching this schema:
{{
  "task_verbs": [
    {{
      "verb": "<verb name, e.g., Calculate>",
      "description": "<complete description of what students must do>"
    }},
    {{
      "verb": "<next verb>",
      "description": "<description>"
    }}
  ]
}}

CRITICAL: 
- Extract EVERY task verb you find, even if they're in different formats
- Look carefully for verbs in tables, lists, or paragraphs
- Include the complete description for each verb
- Common task verbs include: Calculate, Compare, Construct, Describe, Determine, Estimate, Explain, Give examples, Identify, Interpret, Justify, Verify
- Do NOT omit any task verbs{chunk_context}"""
    
    else:
        raise ValueError(f"Unknown section type: {section_type}")
    
    # Prepare images for vision model
    image_parts = []
    for img in images:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        image_parts.append({
            "mime_type": "image/png",
            "data": img_bytes.read()
        })
    
    # Retry logic with exponential backoff
    import time
    for attempt in range(max_retries):
        try:
            # Use extraction model (gemini-2.5-pro) for better quality
            response = client.models.generate_content(
                model=MODEL_EXTRACTION,
                contents=[{"parts": [{"text": prompt}] + [{"inline_data": img} for img in image_parts]}],
                config={"response_mime_type": "application/json"}
            )
            
            if not response or not response.text:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"   ⚠️  Empty response, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                print(f"   ❌ {section_name}: Vision model returned empty response after {max_retries} attempts")
                return None
            
            extracted = json.loads(response.text)
            
            # Ensure we return a dictionary with the correct key structure
            # The model might return just a list, so we need to wrap it
            if isinstance(extracted, list):
                # If it's a list, wrap it in a dict with the section_type key
                if section_type == "exam_sections":
                    extracted = {"exam_sections": extracted}
                elif section_type == "task_verbs":
                    extracted = {"task_verbs": extracted}
                elif section_type == "skills":
                    extracted = {"skills": extracted}
                elif section_type == "big_ideas":
                    extracted = {"big_ideas": extracted}
            
            if chunk_idx == 0 or total_chunks == 1:
                print(f"   ✅ {section_name} extracted successfully")
            else:
                print(f"   ✅ Chunk {chunk_idx + 1} extracted successfully")
            return extracted
            
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"   ⚠️  JSON decode error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                if response and response.text:
                    print(f"   Response preview: {response.text[:300]}...")
                time.sleep(wait_time)
                continue
            print(f"   ❌ {section_name}: Invalid JSON response after {max_retries} attempts - {e}")
            if response and response.text:
                print(f"   Response preview: {response.text[:500]}...")
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"   ⚠️  Error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            print(f"   ❌ Error extracting {section_name} after {max_retries} attempts: {e}")
            return None
    
    return None

def _merge_section_results(chunk_results, section_type):
    """Merge results from multiple chunks."""
    if not chunk_results:
        return None
    
    if section_type == "skills":
        # Merge skills arrays, avoiding duplicates
        all_skills = {}
        for chunk in chunk_results:
            if "skills" in chunk:
                for skill in chunk["skills"]:
                    skill_name = skill.get("skill_name")
                    if skill_name not in all_skills:
                        all_skills[skill_name] = skill
                    else:
                        # Merge subskills
                        existing_subskills = {s["subskill_name"]: s for s in all_skills[skill_name].get("subskills", [])}
                        for subskill in skill.get("subskills", []):
                            subskill_name = subskill.get("subskill_name")
                            if subskill_name not in existing_subskills:
                                all_skills[skill_name]["subskills"].append(subskill)
        return {"skills": list(all_skills.values())}
    
    elif section_type == "big_ideas":
        # Merge big ideas, avoiding duplicates
        all_big_ideas = {}
        for chunk in chunk_results:
            if "big_ideas" in chunk:
                for bi in chunk["big_ideas"]:
                    bi_id = bi.get("id")
                    if bi_id and bi_id not in all_big_ideas:
                        all_big_ideas[bi_id] = bi
        return {"big_ideas": list(all_big_ideas.values())}
    
    elif section_type == "exam_sections":
        # Merge exam sections, combining descriptions
        section_map = {}
        for chunk in chunk_results:
            if "exam_sections" in chunk:
                for section in chunk["exam_sections"]:
                    section_id = section.get("section")
                    if section_id not in section_map:
                        section_map[section_id] = section
                    else:
                        # Merge descriptions
                        existing_descriptions = set(section_map[section_id].get("descriptions", []))
                        new_descriptions = section.get("descriptions", [])
                        for desc in new_descriptions:
                            if desc not in existing_descriptions:
                                section_map[section_id]["descriptions"].append(desc)
        return {"exam_sections": list(section_map.values())}
    
    elif section_type == "task_verbs":
        # Merge task verbs, avoiding duplicates
        all_verbs = {}
        for chunk in chunk_results:
            if "task_verbs" in chunk:
                for verb in chunk["task_verbs"]:
                    verb_name = verb.get("verb")
                    if verb_name and verb_name not in all_verbs:
                        all_verbs[verb_name] = verb
        return {"task_verbs": list(all_verbs.values())}
    
    return chunk_results[0] if chunk_results else None

def extract_unit(pdf_path, unit_info, schema_template, max_retries=3):
    """Stage 3: Extract a single unit from PDF with chunking and retry logic."""
    unit_name = unit_info["name"]
    start_page = unit_info["start_page"] - 1  # Convert to 0-based
    end_page = unit_info["end_page"]
    
    print(f"📚 Extracting {unit_name} (pages {start_page+1}-{end_page})")
    
    # Convert unit pages to images
    images = pdf_to_images(pdf_path)
    unit_images = images[start_page:end_page]
    
    if not unit_images:
        print(f"   ⚠️  No images extracted for {unit_name}")
        return None
    
    # Chunk images if needed (Gemini vision limit is 16 images per request)
    if len(unit_images) > MAX_IMAGES_PER_REQUEST:
        print(f"   📦 Unit spans {len(unit_images)} pages, chunking into batches of {MAX_IMAGES_PER_REQUEST}")
        chunks = []
        for i in range(0, len(unit_images), MAX_IMAGES_PER_REQUEST):
            chunks.append(unit_images[i:i + MAX_IMAGES_PER_REQUEST])
        
        # Extract from each chunk and merge
        all_topics = []
        for chunk_idx, chunk_images in enumerate(chunks):
            chunk_start_page = start_page + (chunk_idx * MAX_IMAGES_PER_REQUEST) + 1
            chunk_end_page = min(start_page + ((chunk_idx + 1) * MAX_IMAGES_PER_REQUEST) + 1, end_page)
            print(f"   📄 Processing chunk {chunk_idx + 1}/{len(chunks)} (pages {chunk_start_page}-{chunk_end_page})")
            
            chunk_result = _extract_unit_chunk(
                chunk_images, unit_name, chunk_idx, len(chunks), max_retries
            )
            if chunk_result:
                all_topics.extend(chunk_result.get("topics", []))
        
        # Get unit metadata from first chunk
        first_chunk = _extract_unit_chunk(unit_images[:MAX_IMAGES_PER_REQUEST], unit_name, 0, 1, max_retries)
        if first_chunk:
            return {
                "name": first_chunk.get("name", unit_name),
                "developing_understanding": first_chunk.get("developing_understanding", ""),
                "building_practices": first_chunk.get("building_practices", ""),
                "preparing_for_exam": first_chunk.get("preparing_for_exam", ""),
                "topics": all_topics
            }
        return None
    else:
        # Single chunk extraction
        return _extract_unit_chunk(unit_images, unit_name, 0, 1, max_retries)

def _extract_unit_chunk(images, unit_name, chunk_idx, total_chunks, max_retries):
    """Extract a single chunk of unit images."""
    
    prompt = f"""Extract ALL unit data from these pages of the AP CED PDF for the unit "{unit_name}".

You MUST extract:
- Unit name (exact name from the PDF)
- Developing Understanding (complete text describing what students should understand - this is usually a full paragraph)
- Building Practices (complete text describing practices/skills emphasized - this is usually a full paragraph)
- Preparing for Exam (complete text describing how this prepares for exam - this is usually a full paragraph)
- ALL Topics (each topic should include):
  - Topic name (exact name from PDF)
  - Big ideas (list of ALL big idea IDs referenced, e.g., ["VAR-1", "UNC-1"])
  - Suggested subskill codes (list of ALL skill codes like ["1.A", "2.B", "2.C"])
  - ALL Learning objectives (each with):
    - ID (format like "VAR-1.A", "UNC-1.B", etc. - extract the exact ID from the PDF)
    - Description (complete description including the skill reference, e.g., "[Skill 1.A]")
    - Essential knowledge (list of ALL knowledge points - these are usually bullet points or numbered items)

Look for:
- Section headers like "Developing Understanding", "Building Practices", "Preparing for Exam"
- Topic headers (usually bold or numbered)
- Learning objective IDs (format like "VAR-1.A", "UNC-1.B")
- Essential knowledge points (usually listed under each learning objective)

Return ONLY valid JSON matching this schema:
{{
  "name": "<unit name>",
  "developing_understanding": "<complete text>",
  "building_practices": "<complete text>",
  "preparing_for_exam": "<complete text>",
  "topics": [
    {{
      "name": "<topic name>",
      "big_ideas": ["<big idea ID>", "<big idea ID>"],
      "suggested_subskill_codes": ["<code>", "<code>"],
      "learning_objectives": [
        {{
          "id": "<LO ID like VAR-1.A>",
          "description": "<complete description>",
          "essential_knowledge": ["<knowledge point 1>", "<knowledge point 2>", "... ALL points ..."]
        }}
      ]
    }}
  ]
}}

CRITICAL: Extract EVERY topic, EVERY learning objective, and EVERY essential knowledge point. Do not omit any."""
    
    # Prepare images
    image_parts = []
    for img in images:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        image_parts.append({
            "mime_type": "image/png",
            "data": img_bytes.read()
        })
    
    # Retry logic with exponential backoff
    import time
    for attempt in range(max_retries):
        try:
            # Use extraction model (gemini-2.5-pro) for better quality
            response = client.models.generate_content(
                model=MODEL_EXTRACTION,
                contents=[{"parts": [{"text": prompt}] + [{"inline_data": img} for img in image_parts]}],
                config={"response_mime_type": "application/json"}
            )
            
            if not response or not response.text:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"   ⚠️  Empty response, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                print(f"   ❌ {unit_name}: Vision model returned empty response after {max_retries} attempts")
                return None
            
            unit_data = json.loads(response.text)
            if chunk_idx == 0 or total_chunks == 1:
                print(f"   ✅ {unit_name} extracted: {len(unit_data.get('topics', []))} topics")
            else:
                print(f"   ✅ Chunk {chunk_idx + 1} extracted: {len(unit_data.get('topics', []))} topics")
            return unit_data
            
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"   ⚠️  JSON decode error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                if response and response.text:
                    print(f"   Response preview: {response.text[:300]}...")
                time.sleep(wait_time)
                continue
            print(f"   ❌ {unit_name}: Invalid JSON response after {max_retries} attempts - {e}")
            if response and response.text:
                print(f"   Response preview: {response.text[:500]}...")
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"   ⚠️  Error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            print(f"   ❌ Error extracting {unit_name} after {max_retries} attempts: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return None

def normalize_to_schema(extracted_data, course_id, course_name):
    """Normalize extracted data to unified schema."""
    normalized = {
        "course_metadata": {
            "course_id": course_id,
            "name": course_name or course_id.replace("ap_", "").replace("_", " ").title(),
            "extraction_method": "vision_model",
            "extraction_date": datetime.now().isoformat()
        },
        "skills": extracted_data.get("skills", []),
        "big_ideas": extracted_data.get("big_ideas", []),
        "units": extracted_data.get("units", []),
        "exam_sections": extracted_data.get("exam_sections", []),
        "task_verbs": extracted_data.get("task_verbs", [])
    }
    
    return normalized

def validate_content(content, schema):
    """Validate content against schema."""
    try:
        validate(instance=content, schema=schema)
        return True, None
    except ValidationError as e:
        return False, str(e)

def extract_ced(course_id):
    """Main extraction function."""
    print(f"\n{'='*60}")
    print(f"Extracting CED for: {course_id}")
    print(f"{'='*60}\n")
    
    # Load config
    config = load_extraction_config(course_id)
    pdf_path = config["pdf_file"]
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Load schema
    schema = load_schema()
    
    # Stage 1: Analyze PDF structure
    structure = analyze_pdf_structure(pdf_path, config)
    
    # Stage 2: Extract sections
    extracted_data = {}
    
    # Extract skills
    if structure.get("skills"):
        skills_data = extract_section(
            pdf_path, "Skills", structure["skills"], "skills", schema
        )
        if skills_data:
            # Handle case where skills_data might be a list directly
            if isinstance(skills_data, list):
                extracted_data["skills"] = skills_data
            else:
                extracted_data["skills"] = skills_data.get("skills", [])
            if not extracted_data["skills"]:
                print(f"   ⚠️  WARNING: Skills section was extracted but is empty!")
        else:
            print(f"   ⚠️  WARNING: Failed to extract skills section")
    else:
        print(f"   ⚠️  WARNING: Skills section not found in PDF structure")
    
    # Extract big ideas
    if structure.get("big_ideas"):
        big_ideas_data = extract_section(
            pdf_path, "Big Ideas", structure["big_ideas"], "big_ideas", schema
        )
        if big_ideas_data:
            # Handle case where big_ideas_data might be a list directly
            if isinstance(big_ideas_data, list):
                extracted_data["big_ideas"] = big_ideas_data
            else:
                extracted_data["big_ideas"] = big_ideas_data.get("big_ideas", [])
            if not extracted_data["big_ideas"]:
                print(f"   ⚠️  WARNING: Big Ideas section was extracted but is empty!")
        else:
            print(f"   ⚠️  WARNING: Failed to extract big ideas section")
    else:
        print(f"   ⚠️  WARNING: Big Ideas section not found in PDF structure")
    
    # Extract exam sections
    if structure.get("exam_sections"):
        print(f"\n📋 Extracting Exam Sections...")
        exam_data = extract_section(
            pdf_path, "Exam Sections", structure["exam_sections"], "exam_sections", schema, max_retries=5
        )
        if exam_data:
            # Handle case where exam_data might be a list directly
            if isinstance(exam_data, list):
                extracted_data["exam_sections"] = exam_data
            else:
                extracted_data["exam_sections"] = exam_data.get("exam_sections", [])
            if not extracted_data["exam_sections"]:
                print(f"   ⚠️  WARNING: Exam Sections section was extracted but is empty!")
                print(f"   🔄 Attempting to re-extract with expanded page range...")
                # Try expanding the page range slightly
                expanded_range = {
                    "start_page": max(1, structure["exam_sections"]["start_page"] - 2),
                    "end_page": min(len(pdf_to_images(pdf_path)), structure["exam_sections"]["end_page"] + 5)
                }
                exam_data_retry = extract_section(
                    pdf_path, "Exam Sections", expanded_range, "exam_sections", schema, max_retries=3
                )
                if exam_data_retry:
                    if isinstance(exam_data_retry, list):
                        extracted_data["exam_sections"] = exam_data_retry
                    elif exam_data_retry.get("exam_sections"):
                        extracted_data["exam_sections"] = exam_data_retry.get("exam_sections", [])
                    print(f"   ✅ Re-extraction successful!")
            else:
                # Check if descriptions are missing
                for section in extracted_data["exam_sections"]:
                    desc_count = len(section.get("descriptions", []))
                    if desc_count < 2:
                        print(f"   ⚠️  WARNING: Section {section.get('section')} has only {desc_count} description(s)!")
                    else:
                        print(f"   ✅ Section {section.get('section')}: {desc_count} descriptions extracted")
        else:
            print(f"   ⚠️  WARNING: Failed to extract exam sections")
    else:
        print(f"   ⚠️  WARNING: Exam Sections section not found in PDF structure")
        print(f"   💡 Tip: Exam sections are often near the end of the PDF")
    
    # Extract task verbs (optional but important for FRQs)
    task_verbs_extracted = False
    if structure.get("task_verbs"):
        print(f"\n📝 Extracting Task Verbs...")
        task_verbs_data = extract_section(
            pdf_path, "Task Verbs", structure["task_verbs"], "task_verbs", schema, max_retries=5
        )
        if task_verbs_data:
            # Handle case where task_verbs_data might be a list directly
            if isinstance(task_verbs_data, list):
                extracted_data["task_verbs"] = task_verbs_data
            else:
                extracted_data["task_verbs"] = task_verbs_data.get("task_verbs", [])
            if not extracted_data["task_verbs"]:
                print(f"   ⚠️  WARNING: Task Verbs section was extracted but is empty!")
                print(f"   🔄 Attempting to re-extract with expanded page range...")
                # Try expanding the page range slightly
                all_images = pdf_to_images(pdf_path)
                expanded_range = {
                    "start_page": max(1, structure["task_verbs"]["start_page"] - 5),
                    "end_page": min(len(all_images), structure["task_verbs"]["end_page"] + 5)
                }
                task_verbs_data_retry = extract_section(
                    pdf_path, "Task Verbs", expanded_range, "task_verbs", schema, max_retries=3
                )
                if task_verbs_data_retry:
                    if isinstance(task_verbs_data_retry, list):
                        extracted_data["task_verbs"] = task_verbs_data_retry
                    elif task_verbs_data_retry.get("task_verbs"):
                        extracted_data["task_verbs"] = task_verbs_data_retry.get("task_verbs", [])
                    if extracted_data.get("task_verbs"):
                        print(f"   ✅ Re-extraction successful!")
                        task_verbs_extracted = True
            else:
                print(f"   ✅ Task Verbs: {len(extracted_data['task_verbs'])} verbs extracted")
                task_verbs_extracted = True
        else:
            print(f"   ⚠️  WARNING: Failed to extract task verbs section")
    
    # Fallback: Search entire PDF for task verbs if not found
    if not task_verbs_extracted:
        print(f"\n📝 Task Verbs not found in structure analysis. Searching entire PDF...")
        all_images = pdf_to_images(pdf_path)
        total_pages = len(all_images)
        
        # Strategy 1: Search exam sections area (task verbs often near exam info)
        if structure.get("exam_sections"):
            exam_start = structure["exam_sections"]["start_page"] - 1
            exam_end = min(structure["exam_sections"]["end_page"] + 10, total_pages)
            print(f"   🔍 Strategy 1: Searching exam sections area (pages {exam_start+1}-{exam_end})...")
            exam_area_range = {"start_page": exam_start + 1, "end_page": exam_end}
            task_verbs_data = extract_section(
                pdf_path, "Task Verbs (exam area)", exam_area_range, "task_verbs", schema, max_retries=3
            )
            if task_verbs_data:
                if isinstance(task_verbs_data, list):
                    extracted_data["task_verbs"] = task_verbs_data
                elif task_verbs_data.get("task_verbs"):
                    extracted_data["task_verbs"] = task_verbs_data.get("task_verbs", [])
                if extracted_data.get("task_verbs"):
                    print(f"   ✅ Found task verbs in exam area: {len(extracted_data['task_verbs'])} verbs")
                    task_verbs_extracted = True
        
        # Strategy 2: Search last 30 pages (task verbs often in appendix)
        if not task_verbs_extracted:
            last_pages_start = max(1, total_pages - 30)
            print(f"   🔍 Strategy 2: Searching last 30 pages (pages {last_pages_start}-{total_pages})...")
            last_pages_range = {"start_page": last_pages_start, "end_page": total_pages}
            task_verbs_data = extract_section(
                pdf_path, "Task Verbs (end of PDF)", last_pages_range, "task_verbs", schema, max_retries=3
            )
            if task_verbs_data:
                if isinstance(task_verbs_data, list):
                    extracted_data["task_verbs"] = task_verbs_data
                elif task_verbs_data.get("task_verbs"):
                    extracted_data["task_verbs"] = task_verbs_data.get("task_verbs", [])
                if extracted_data.get("task_verbs"):
                    print(f"   ✅ Found task verbs at end of PDF: {len(extracted_data['task_verbs'])} verbs")
                    task_verbs_extracted = True
        
        # Strategy 3: Search entire PDF in chunks (last resort)
        if not task_verbs_extracted:
            print(f"   🔍 Strategy 3: Searching entire PDF in chunks...")
            # Sample pages throughout the PDF
            sample_pages = []
            step = max(1, total_pages // 20)  # Sample ~20 pages
            for i in range(0, total_pages, step):
                sample_pages.append(i)
            
            # Search in batches
            for batch_start in range(0, len(sample_pages), 10):
                batch_pages = sample_pages[batch_start:batch_start+10]
                if not batch_pages:
                    continue
                
                batch_range = {
                    "start_page": batch_pages[0] + 1,
                    "end_page": batch_pages[-1] + 1
                }
                print(f"   🔍 Searching pages {batch_range['start_page']}-{batch_range['end_page']}...")
                task_verbs_data = extract_section(
                    pdf_path, f"Task Verbs (pages {batch_range['start_page']}-{batch_range['end_page']})", 
                    batch_range, "task_verbs", schema, max_retries=2
                )
                if task_verbs_data:
                    if isinstance(task_verbs_data, list):
                        extracted_data["task_verbs"] = task_verbs_data
                    elif task_verbs_data.get("task_verbs"):
                        extracted_data["task_verbs"] = task_verbs_data.get("task_verbs", [])
                    if extracted_data.get("task_verbs"):
                        print(f"   ✅ Found task verbs: {len(extracted_data['task_verbs'])} verbs")
                        task_verbs_extracted = True
                        break
        
        if not task_verbs_extracted:
            print(f"   ⚠️  WARNING: Task verbs not found after exhaustive search")
            print(f"   💡 Task verbs may not be present in this CED, or may be embedded in exam sections")
    
    # Stage 3: Extract units
    units = []
    if structure.get("units"):
        units_to_extract = structure["units"]
        
        # TEST MODE: Only extract first unit
        if TEST_MODE:
            print(f"\n🧪 TEST MODE: Extracting only first unit (set TEST_MODE=False to extract all)")
            units_to_extract = units_to_extract[:1]
        
        for unit_info in units_to_extract:
            unit_data = extract_unit(pdf_path, unit_info, schema, max_retries=3)
            if unit_data:
                units.append(unit_data)
    
    extracted_data["units"] = units
    
    # Normalize to schema
    course_name = config.get("course_name") or course_id.replace("ap_", "").replace("_", " ").title()
    normalized = normalize_to_schema(extracted_data, course_id, course_name)
    
    # Validate
    is_valid, error = validate_content(normalized, schema)
    if not is_valid:
        print(f"\n⚠️  Validation warnings: {error}")
        print("   Content extracted but may need manual review")
    else:
        print(f"\n✅ Content validated against schema")
    
    # Save
    os.makedirs(CONTENT_DIR, exist_ok=True)
    output_path = os.path.join(CONTENT_DIR, f"{course_id}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Extraction complete!")
    print(f"   Saved to: {output_path}")
    print(f"   Skills: {len(normalized['skills'])} categories")
    print(f"   Big Ideas: {len(normalized['big_ideas'])}")
    print(f"   Units: {len(normalized['units'])}")
    print(f"   Exam Sections: {len(normalized['exam_sections'])}")
    print(f"   Task Verbs: {len(normalized.get('task_verbs', []))}")
    
    return normalized

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        course_id = sys.argv[1]
    else:
        # Default: extract all courses in courses directory
        course_files = [f for f in os.listdir(COURSES_DIR) if f.endswith(".pdf")]
        course_ids = [f.replace(".pdf", "") for f in course_files]
        
        if not course_ids:
            print("No PDF files found in courses/ directory")
            sys.exit(1)
        
        print("Available courses:")
        for i, cid in enumerate(course_ids, 1):
            print(f"  {i}. {cid}")
        
        choice = input("\nEnter course number or course_id: ").strip()
        
        if choice.isdigit():
            course_id = course_ids[int(choice) - 1]
        else:
            course_id = choice
    
    try:
        extract_ced(course_id)
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
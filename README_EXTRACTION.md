# Phase 1: Content Extraction

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install poppler (required for pdf2image):
   - **macOS**: `brew install poppler`
   - **Linux**: `sudo apt-get install poppler-utils`
   - **Windows**: Download from https://github.com/oschwartz10612/poppler-windows/releases

3. Set up environment variable:
```bash
export GEMINI_API_KEY="your-api-key-here"
```
Or create a `.env` file:
```
GEMINI_API_KEY=your-api-key-here
```

## Usage

### Extract a single course:
```bash
python pdf_to_json.py ap_microeconomics
```

### Interactive mode (select from available courses):
```bash
python pdf_to_json.py
```

## What It Does

1. **Stage 1: PDF Structure Analysis**
   - Uses Gemini 2.0 Flash Vision to analyze PDF
   - Identifies page ranges for: Skills, Big Ideas, Units, Exam Sections, Task Verbs

2. **Stage 2: Section Extraction**
   - Extracts each section separately using vision model
   - Skills, Big Ideas, Exam Sections, Task Verbs

3. **Stage 3: Unit Extraction**
   - Extracts each unit separately
   - Includes topics, learning objectives, essential knowledge

4. **Normalization & Validation**
   - Normalizes to unified schema
   - Validates against `utils/schemas/content_schema.json`
   - Saves to `utils/content/{course_id}.json`

## Output

Extracted content is saved to:
```
utils/content/{course_id}.json
```

Contains:
- Course metadata
- Skills (with subskills)
- Big Ideas
- Units (with topics, learning objectives, essential knowledge)
- Exam Sections
- Task Verbs (if present)

## Configuration

Extraction configs are auto-generated in:
```
utils/config/{course_id}_extraction.json
```

You can customize these to provide better hints for section detection.


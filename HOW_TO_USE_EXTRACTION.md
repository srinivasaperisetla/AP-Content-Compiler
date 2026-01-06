# How to Use pdf_to_json.py

## 🎯 What It Does

`pdf_to_json.py` is **Phase 1: Content Extraction** - it extracts structured content from AP Course and Exam Description (CED) PDFs and converts them into standardized JSON format.

### The Process

```
AP Statistics CED PDF (courses/ap_statistics.pdf)
    ↓
[Stage 1: Structure Analysis]
    Uses Gemini 2.0 Flash Vision to analyze PDF
    Finds page ranges for: Skills, Big Ideas, Units, Exam Sections, Task Verbs
    ↓
[Stage 2: Section Extraction]
    Extracts each section separately:
    - Skills (with subskills and codes)
    - Big Ideas (with IDs and descriptions)
    - Exam Sections (MCQ & FRQ requirements)
    - Task Verbs (for FRQ generation)
    ↓
[Stage 3: Unit Extraction]
    Extracts each unit separately:
    - Unit name, developing understanding, building practices
    - Topics with learning objectives
    - Essential knowledge for each learning objective
    ↓
[Normalization & Validation]
    Converts to unified schema
    Validates against content_schema.json
    ↓
Standardized JSON (utils/content/ap_statistics.json)
```

### What Gets Extracted

1. **Skills** → Skill categories with subskills (e.g., "1.A", "2.B")
2. **Big Ideas** → Conceptual themes (e.g., "VAR", "UNC")
3. **Units** → All course units with:
   - Topics
   - Learning objectives (with IDs like "VAR-1.A")
   - Essential knowledge
4. **Exam Sections** → MCQ and FRQ requirements
5. **Task Verbs** → Verbs used in FRQs (e.g., "Calculate", "Justify")

---

## 📋 How to Use for AP Statistics

### Prerequisites

1. **Install dependencies:**
```bash
pip install google-genai python-dotenv jsonschema pdf2image Pillow
```

2. **Install poppler** (required for PDF to image conversion):
   - **macOS**: `brew install poppler`
   - **Linux**: `sudo apt-get install poppler-utils`
   - **Windows**: Download from https://github.com/oschwartz10612/poppler-windows/releases

3. **Set up API key:**
   - Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your-api-key-here
   ```
   - Or export as environment variable:
   ```bash
   export GEMINI_API_KEY=your-api-key-here
   ```

### Usage

#### **Option 1: Direct Command (Recommended)**

```bash
python pdf_to_json.py ap_statistics
```

This will:
1. Look for `courses/ap_statistics.pdf`
2. Auto-generate extraction config if needed (`utils/config/ap_statistics_extraction.json`)
3. Extract all content
4. Save to `utils/content/ap_statistics.json`

#### **Option 2: Interactive Mode**

```bash
python pdf_to_json.py
```

This will:
1. Show list of all available PDFs in `courses/` directory
2. Let you select by number or type course ID
3. Extract the selected course

---

## 🔍 Step-by-Step: What Happens When You Run It

### Step 1: Configuration
```
Script checks for: utils/config/ap_statistics_extraction.json
If not found → Creates default config with section hints
```

### Step 2: PDF Structure Analysis
```
📊 Analyzing PDF structure: courses/ap_statistics.pdf
   Uses Gemini 2.0 Flash Vision to analyze first 10 pages
   Identifies where sections are located:
   - Skills: pages 5-12
   - Big Ideas: pages 13-18
   - Units: Unit 1 (pages 20-45), Unit 2 (pages 46-70), etc.
   - Exam Sections: pages 200-220
   - Task Verbs: pages 15-16
```

### Step 3: Section Extraction
```
📄 Extracting Skills (pages 5-12)
   ✅ Skills extracted successfully

📄 Extracting Big Ideas (pages 13-18)
   ✅ Big Ideas extracted successfully

📄 Extracting Exam Sections (pages 200-220)
   ✅ Exam Sections extracted successfully

📄 Extracting Task Verbs (pages 15-16)
   ✅ Task Verbs extracted successfully
```

### Step 4: Unit Extraction
```
📚 Extracting Unit 1: Exploring One-Variable Data (pages 20-45)
   ✅ Unit 1 extracted: 10 topics

📚 Extracting Unit 2: Exploring Two-Variable Data (pages 46-70)
   ✅ Unit 2 extracted: 8 topics

... (continues for all units)
```

### Step 5: Normalization & Validation
```
✅ Content validated against schema

✅ Extraction complete!
   Saved to: utils/content/ap_statistics.json
   Skills: 4 categories
   Big Ideas: 3
   Units: 9
   Exam Sections: 2
   Task Verbs: 12
```

---

## 📁 Output

### Generated Files

1. **Extraction Config** (auto-generated if missing):
   - `utils/config/ap_statistics_extraction.json`
   - Contains section detection hints and settings

2. **Extracted Content**:
   - `utils/content/ap_statistics.json`
   - Contains all extracted data in standardized format

### Output Structure

```json
{
  "course_metadata": {
    "course_id": "ap_statistics",
    "name": "AP Statistics",
    "extraction_method": "vision_model",
    "extraction_date": "2024-01-15T20:30:00"
  },
  "skills": [
    {
      "skill_name": "Data Analysis",
      "skill_description": "...",
      "subskills": [
        {
          "subskill_name": "2.A",
          "subskill_description": "..."
        }
      ]
    }
  ],
  "big_ideas": [...],
  "units": [
    {
      "name": "Exploring One-Variable Data",
      "developing_understanding": "...",
      "topics": [
        {
          "name": "Topic name",
          "learning_objectives": [
            {
              "id": "VAR-1.A",
              "description": "...",
              "essential_knowledge": ["..."]
            }
          ]
        }
      ]
    }
  ],
  "exam_sections": [...],
  "task_verbs": [...]
}
```

---

## ⚙️ Configuration

### Customizing Extraction

You can customize the extraction by editing:
```
utils/config/ap_statistics_extraction.json
```

**Example customization:**
```json
{
  "course_id": "ap_statistics",
  "pdf_file": "courses/ap_statistics.pdf",
  "section_guide": {
    "skills": {
      "hints": ["Skills", "Science Practices"],
      "required": true
    },
    "big_ideas": {
      "hints": ["Big Ideas", "Themes"],
      "required": true
    }
  }
}
```

**Why customize?**
- If vision model misses a section, add more specific hints
- If sections are in unusual locations, you can provide page hints
- Adjust detection method if needed

---

## 🐛 Troubleshooting

### Error: "PDF not found"
- **Solution**: Make sure `courses/ap_statistics.pdf` exists
- Check the filename matches exactly (case-sensitive)

### Error: "Failed to convert PDF to images"
- **Solution**: Install poppler (see Prerequisites)
- On macOS: `brew install poppler`
- Verify: `which pdftoppm` should return a path

### Error: "Missing GEMINI_API_KEY"
- **Solution**: Set API key in `.env` file or environment variable
- Create `.env` file: `GEMINI_API_KEY=your-key`

### Error: "Section not found"
- **Solution**: The vision model couldn't find the section
- Edit `utils/config/ap_statistics_extraction.json` to add more hints
- Or manually specify page ranges if you know them

### Validation Warnings
- **What it means**: Extracted content doesn't fully match schema
- **Solution**: Review the extracted JSON, may need manual correction
- Check that all required fields are present

---

## 📊 Example: Running for AP Statistics

```bash
$ python pdf_to_json.py ap_statistics

============================================================
Extracting CED for: ap_statistics
============================================================

📊 Analyzing PDF structure: courses/ap_statistics.pdf
✅ PDF structure analyzed
   Skills: pages 5-12
   Big Ideas: pages 13-18
   Units: 9 units found
   Exam Sections: pages 200-220

📄 Extracting Skills (pages 5-12)
   ✅ Skills extracted successfully

📄 Extracting Big Ideas (pages 13-18)
   ✅ Big Ideas extracted successfully

📄 Extracting Exam Sections (pages 200-220)
   ✅ Exam Sections extracted successfully

📄 Extracting Task Verbs (pages 15-16)
   ✅ Task Verbs extracted successfully

📚 Extracting Unit 1: Exploring One-Variable Data (pages 20-45)
   ✅ Unit 1 extracted: 10 topics

📚 Extracting Unit 2: Exploring Two-Variable Data (pages 46-70)
   ✅ Unit 2 extracted: 8 topics

... (continues for all 9 units)

✅ Content validated against schema

✅ Extraction complete!
   Saved to: utils/content/ap_statistics.json
   Skills: 4 categories
   Big Ideas: 3
   Units: 9
   Exam Sections: 2
   Task Verbs: 12
```

---

## 🎯 What Happens Next?

After extraction, you'll have:
- ✅ `utils/content/ap_statistics.json` - Ready for question generation
- ✅ All skills, big ideas, units, learning objectives extracted
- ✅ Validated against schema

**Next step**: Use this JSON in Phase 2 (MCQ/FRQ generation) to create practice questions!

---

## 💡 Tips

1. **First run**: Let it auto-generate the config, then customize if needed
2. **Large PDFs**: May take 5-10 minutes depending on PDF size
3. **API costs**: Uses Gemini API - check your usage
4. **Validation**: Always check validation output - warnings mean review needed
5. **Re-running**: Safe to re-run - will overwrite existing JSON

---

## 📝 Summary

**Command**: `python pdf_to_json.py ap_statistics`

**Input**: `courses/ap_statistics.pdf`

**Output**: `utils/content/ap_statistics.json`

**What it does**: Extracts all course content (skills, units, learning objectives, etc.) from PDF using vision AI

**Time**: ~5-10 minutes for a typical CED

**Result**: Standardized JSON ready for question generation!


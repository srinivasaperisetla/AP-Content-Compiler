# Complete Guide: AP Content Compiler System

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Complete Pipeline](#complete-pipeline)
3. [Content Extraction](#content-extraction)
4. [Content Standardization](#content-standardization)
5. [MCQ Generation](#mcq-generation)
6. [FRQ Generation](#frq-generation)
7. [Stimulus Generation (Images, SVG, Charts)](#stimulus-generation)
8. [Validation & Quality Assurance](#validation--quality-assurance)
9. [Models & Technology](#models--technology)
10. [Plan Moving Forward](#plan-moving-forward)

---

## 🎯 System Overview

### Purpose

Generate high-quality AP practice questions (MCQs and FRQs) for all AP courses by extracting content from College Board Course and Exam Descriptions (CEDs) and using AI to create aligned, authentic practice questions.

### Core Challenge

- **PDF Variance**: Each AP course CED has different internal structure and layout within a single PDF
- **Course Variance**: Different courses need different question formats, stimulus types, and requirements
- **Quality**: Questions must align to learning objectives, assess correct skills, and match exam style

### Solution Architecture

```
Single CED PDF → Vision Model Extraction → Standardized JSON → Question Generation → HTML Output
```

**Key Innovation**: Config-driven system that uses vision models to handle PDF variance without manual sectioning.

---

## 🔄 Complete Pipeline

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: CONTENT EXTRACTION                                  │
│  Input: Single CED PDF                                       │
│  Output: Standardized JSON                                   │
│  Validation: Schema validation, content checks              │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: CONTENT PREPARATION                                 │
│  Input: Standardized JSON + Course Config                    │
│  Output: Unit Payloads (ready for generation)               │
│  Validation: Payload completeness, alignment checks         │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: QUESTION GENERATION                                 │
│  Input: Unit Payloads + Generation Config                   │
│  Output: Question JSON (MCQs and FRQs)                      │
│  Validation: Schema, alignment, quality checks               │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: POST-PROCESSING & OUTPUT                           │
│  Input: Question JSON                                        │
│  Output: HTML question sets                                  │
│  Validation: Final quality checks, rendering validation     │
└─────────────────────────────────────────────────────────────┘
```

### Detailed Step-by-Step Process

#### **Step 1: Load Extraction Config**

- **Input**: Course ID (e.g., "ap_microeconomics")
- **Process**: Load `utils/config/{course_id}_extraction.json`
- **Output**: PDF file path and section detection hints
- **Validation**: Config file exists, required fields present

#### **Step 2: Analyze PDF Structure**

- **Input**: Single CED PDF + Extraction config
- **Process**: Use vision model (Gemini 2.0 Pro Vision) to analyze PDF and identify section boundaries
- **Output**: Section locations (skills, big ideas, units, exam sections, task verbs)
- **Validation**: All required sections found, page ranges valid

#### **Step 3: Extract Content from Sections**

- **Input**: PDF + Section locations
- **Process**: Extract each section separately using vision model
- **Output**: Raw extracted content (may be course-specific format)
- **Validation**: All required sections extracted, no missing data

#### **Step 4: Normalize to Unified Schema**

- **Input**: Raw extracted content
- **Process**: Convert to standardized JSON schema
- **Output**: `utils/content/{course_id}.json`
- **Validation**: Schema validation, required fields present

#### **Step 5: Load Course Config**

- **Input**: Course ID
- **Process**: Load `utils/config/{course_id}.json`
- **Output**: Generation parameters (stimulus types, num choices, etc.)
- **Validation**: Config valid, all required fields present

#### **Step 6: Build Unit Payloads**

- **Input**: Standardized JSON + Course config
- **Process**: Extract unit-specific data (LOs, skills, big ideas, exam context)
- **Output**: Unit payloads ready for generation
- **Validation**: All required fields in payload, learning objectives valid

#### **Step 7: Generate MCQs**

- **Input**: Unit payload + Course config
- **Process**: Batch generation with validation
- **Output**: MCQ JSON
- **Validation**: Schema, alignment, quality checks

#### **Step 8: Generate FRQs**

- **Input**: Unit payload (with task verbs) + Course config
- **Process**: One-by-one generation with part coherence validation
- **Output**: FRQ JSON
- **Validation**: Schema, part coherence, task verb validation

#### **Step 9: Post-Process & Render**

- **Input**: Question JSON
- **Process**: Format stimuli, render HTML
- **Output**: HTML question sets
- **Validation**: HTML valid, all questions render correctly

---

## 📄 Content Extraction

### The Approach

All AP courses use **single PDF files** containing the entire CED. The challenge is that each PDF has different internal structure and layout. We use **vision models** to automatically detect and extract sections.

### Solution: Vision Model-Based Extraction

#### **Extraction Config Structure**

```json
{
  "course_id": "ap_microeconomics",
  "pdf_file": "ap_specs/ap_microeconomics.pdf",
  "extraction_method": "vision_model",
  "model": "gemini-2.0-pro-vision",
  "section_guide": {
    "skills": {
      "detection_method": "auto",
      "hints": ["Skills", "Science Practices", "Course Skills"],
      "required": true
    },
    "big_ideas": {
      "detection_method": "auto",
      "hints": ["Big Ideas", "Themes", "Enduring Understandings"],
      "required": true
    },
    "units": {
      "detection_method": "auto",
      "hints": ["Unit 1", "Unit 2", "Course Units", "Module"],
      "required": true,
      "extract_each_unit_separately": true
    },
    "exam_sections": {
      "detection_method": "auto",
      "hints": [
        "Exam Information",
        "Section I",
        "Section II",
        "Multiple Choice",
        "Free Response"
      ],
      "required": true
    },
    "task_verbs": {
      "detection_method": "auto",
      "hints": ["Task Verbs", "Command Terms", "Action Verbs"],
      "required": false
    }
  },
  "special_handling": {
    "has_formulas": true,
    "has_diagrams": true,
    "has_tables": true,
    "has_passages": false,
    "has_historical_images": false
  }
}
```

### Extraction Process

#### **Stage 1: PDF Structure Analysis**

```
1. Load PDF file
2. Convert PDF pages to images (or use vision API directly)
3. Use vision model to analyze entire PDF structure:
   - "Identify page ranges for: Skills, Big Ideas, Units, Exam Sections, Task Verbs"
   - Vision model returns JSON with section boundaries
4. Validate section detection (all required sections found)
```

**Example Vision Model Prompt:**

```
Analyze this AP Microeconomics CED PDF.

Identify the page ranges for:
1. Skills section (with subskills and codes)
2. Big Ideas/Themes section
3. Course Units (list all units and their page ranges)
4. Exam Information (MCQ and FRQ sections)
5. Task Verbs (if present)

Return JSON:
{
  "skills": {"start_page": 5, "end_page": 12},
  "big_ideas": {"start_page": 13, "end_page": 18},
  "units": [
    {"name": "Unit 1: Basic Economic Concepts", "start_page": 20, "end_page": 45},
    {"name": "Unit 2: Supply and Demand", "start_page": 46, "end_page": 70},
    ...
  ],
  "exam_sections": {"start_page": 200, "end_page": 220},
  "task_verbs": {"start_page": 15, "end_page": 16}
}
```

#### **Stage 2: Section-by-Section Extraction**

```
For each section (skills, big ideas, exam sections, task verbs):
1. Extract pages for this section (using page ranges from Stage 1)
2. Build extraction prompt based on section type
3. Use vision model to extract structured data
4. Validate extracted data against schema
5. Normalize format
```

**Example: Skills Extraction**

```
Extract skills from pages 5-12 of this AP Microeconomics CED.

Extract:
- Skill categories (e.g., "Analyze Economic Concepts")
- Subskills with codes (e.g., "1.A", "1.B")
- Skill descriptions

Return JSON matching this schema:
{
  "skills": [
    {
      "skill_name": "...",
      "skill_description": "...",
      "subskills": [
        {
          "subskill_name": "1.A",
          "subskill_description": "..."
        }
      ]
    }
  ]
}
```

#### **Stage 3: Unit Extraction (Special Handling)**

```
For each unit (identified in Stage 1):
1. Extract pages for this unit
2. Use vision model to extract:
   - Unit name
   - Developing Understanding
   - Building Practices
   - Preparing for Exam
   - Topics (with learning objectives, essential knowledge)
3. Validate unit data
4. Normalize to schema
```

**Example: Unit Extraction**

```
Extract Unit 1 data from pages 20-45 of this AP Microeconomics CED.

Extract:
- Unit name
- Developing Understanding
- Building Practices
- Preparing for Exam
- Topics (each with):
  - Topic name
  - Big ideas
  - Suggested subskill codes
  - Learning objectives (with IDs, descriptions, essential knowledge)

Return JSON matching this schema:
{
  "name": "...",
  "developing_understanding": "...",
  "building_practices": "...",
  "preparing_for_exam": "...",
  "topics": [...]
}
```

### What Gets Extracted

#### **1. Skills Section** (CRITICAL)

- Skill categories (e.g., "Data Analysis")
- Subskills with codes (e.g., "2.A", "2.B")
- Skill descriptions
- **Why Important**: Questions must assess these skills

#### **2. Big Ideas Section** (IMPORTANT)

- Big idea IDs (e.g., "VAR", "UNC")
- Big idea names and descriptions
- **Why Important**: Questions should connect to conceptual themes

#### **3. Units Section** (MOST CRITICAL)

- Unit names
- Topics within units
- **Learning objectives** (with IDs like "VAR-1.A")
- **Essential knowledge** for each learning objective
- Developing understanding, building practices, preparing for exam
- **Why Important**: Questions MUST align to learning objectives

#### **4. Exam Sections** (IMPORTANT)

- Section I (MCQ) requirements: number of questions, timing, weighting
- Section II (FRQ) requirements: number of questions, timing, weighting
- Question style descriptions
- **Why Important**: Questions must match exam format and style

#### **5. Task Verbs** (CRITICAL FOR FRQ)

- Task verbs (e.g., "Calculate", "Justify", "Explain")
- Verb descriptions (what students must do)
- **Why Important**: FRQs must use correct task verbs appropriately

### Extraction Validation

**At Each Step:**

1. **File Existence**: PDF file exists
2. **PDF Structure Analysis**: Vision model successfully identifies sections
3. **Section Detection**: All required sections found
4. **Page Ranges**: Valid page ranges for each section
5. **Content Extraction**: Text/data extracted successfully (not empty)
6. **Schema Validation**: Extracted JSON matches schema
7. **Completeness**: All required fields present
8. **Learning Objective IDs**: Valid format (e.g., "VAR-1.A")
9. **Skill Codes**: Valid format (e.g., "1.A", "2.B")

---

## 📐 Content Standardization

### Unified Schema

All courses use the same JSON structure:

```json
{
  "course_metadata": {
    "course_id": "ap_microeconomics",
    "name": "AP Microeconomics",
    "version": "2024-2025",
    "extraction_method": "vision_model",
    "extraction_date": "2024-01-15"
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
  "big_ideas": [
    {
      "id": "VAR",
      "name": "VARIATION AND DISTRIBUTION",
      "description": "..."
    }
  ],
  "units": [
    {
      "name": "Basic Economic Concepts",
      "developing_understanding": "...",
      "building_practices": "...",
      "preparing_for_exam": "...",
      "topics": [
        {
          "name": "Topic name",
          "big_ideas": ["VAR-1"],
          "suggested_subskill_codes": ["1.A", "2.B"],
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
  "exam_sections": [
    {
      "section": "I",
      "question_type": "Multiple-choice questions",
      "Number of Questions": "60",
      "exam_weighting": "66.67%",
      "timing": "70 minutes",
      "descriptions": ["..."]
    }
  ],
  "task_verbs": [
    {
      "verb": "Calculate",
      "description": "..."
    }
  ]
}
```

### Normalization Process

1. **Schema Validation**: Ensure extracted data matches schema
2. **Field Mapping**: Map course-specific fields to standard fields
3. **Format Normalization**: Standardize IDs, codes, formatting
4. **Enrichment**: Add cross-references, metadata
5. **Final Validation**: Complete schema check

---

## 🎲 MCQ Generation

### Input Requirements

```python
{
    "course_config": {...},        # From utils/config/{course_id}.json
    "unit_payload": {...},         # Built from standardized JSON
    "generation_params": {
        "num_questions": 25,
        "batch_size": 5,
        "difficulty_distribution": {"easy": 0.3, "medium": 0.5, "hard": 0.2}
    }
}
```

### Generation Process

#### **Step 1: Context Assembly**

```
Load:
- Course config (stimulus types, num choices, batch size)
- Unit payload (learning objectives, skills, big ideas, exam context)
- Prompt template
- JSON schema
```

#### **Step 2: Prompt Construction**

```
Build prompt with:
- Unit context (name, developing understanding, preparing for exam)
- Learning objectives (IDs and descriptions)
- Skills (codes and definitions)
- Big ideas (IDs and descriptions)
- Exam section context (MCQ requirements)
- Generation requirements (num questions, num choices, difficulty)
- Stimulus requirements (allowed types)
```

#### **Step 3: Batch Generation**

```
Loop until target count reached:
1. Generate batch of N questions (default: 5)
2. Validate each question:
   - Schema validation
   - Content validation (not empty, proper format)
   - Alignment validation (learning objectives exist)
   - Skill validation (skill codes valid)
3. Check for duplicates
4. If valid, add to collection
5. If invalid, retry (up to 3 times)
```

#### **Step 4: Post-Processing**

```
For each question:
1. Normalize choice labels (strip "A.", "B.", etc.)
2. Process stimulus:
   - If table: Convert markdown to HTML
   - If SVG: Validate and escape
   - If image: Validate URL or generate
3. Add metadata (timestamp, course, model version)
```

### MCQ Validation

**Schema Validation:**

- Matches JSON schema exactly
- Required fields present
- Correct data types

**Content Validation:**

- Question text not empty (min 10 characters)
- Correct number of choices (4 or 5, per config)
- Valid correct_choice_index
- Explanation present and clear

**Alignment Validation:**

- At least one learning objective aligned
- Learning objective IDs exist in unit
- Skill codes exist in unit
- Big ideas exist in course

**Quality Validation:**

- Difficulty level valid (easy/medium/hard)
- Stimulus (if present) is valid type
- Question depends on stimulus (if present)
- Distractors reflect misconceptions

---

## 📝 FRQ Generation

### Input Requirements

```python
{
    "course_config": {...},
    "unit_payload": {...},  # Includes task_verbs
    "generation_params": {
        "num_frqs": 25,
        "batch_size": 1,  # FRQs are complex, one at a time
        "min_parts": 3,
        "max_parts": 5
    }
}
```

### Generation Process

#### **Step 1: Context Assembly (FRQ-Specific)**

```
Load:
- Course config
- Unit payload (includes task_verbs - CRITICAL for FRQ)
- FRQ prompt template
- JSON schema
```

#### **Step 2: Prompt Construction**

```
Build prompt with:
- Unit context
- Learning objectives
- Skills
- Task verbs (with descriptions) - FRQ-specific!
- Exam section context (FRQ requirements)
- Part structure guidelines
- Scoring guideline requirements
```

#### **Step 3: FRQ Generation (One at a Time)**

```
For each FRQ:
1. Generate contextual scenario
2. Generate multiple parts (a, b, c, ...):
   - Each part has label, prompt, task verb, point value
3. Generate scoring guidelines
4. Add stimulus if needed
5. Validate:
   - Schema validation
   - Part coherence (parts build on each other)
   - Task verb appropriateness
   - Scoring guideline completeness
6. If invalid, retry (up to 3 times)
```

#### **Step 4: Post-Processing**

```
For each FRQ:
1. Format stimulus (if present)
2. Format scoring guidelines
3. Ensure part labels correct (a, b, c, ...)
4. Add metadata (timestamp, num parts, total points)
```

### FRQ Validation

**Schema Validation:**

- Matches JSON schema exactly
- Required fields present

**Content Validation:**

- Context/scenario substantial (min 50 words)
- Correct number of parts (3-5, per config)
- Part labels correct (a, b, c, ...)
- Each part has prompt and task verb

**Part Coherence Validation:**

- Parts build on each other logically
- Complexity generally increases
- Part (a) is foundational
- Part (b) builds on (a)
- Part (c) is more complex

**Task Verb Validation:**

- Task verbs are valid (exist in course task verbs)
- Task verbs used appropriately
- "Justify" requires evidence + reasoning
- "Calculate" requires computation

**Scoring Guidelines Validation:**

- Scoring guidelines present (if required)
- Guidelines are clear and point-oriented
- Cover common student errors

---

## 🎨 Stimulus Generation (Images, SVG, Charts, Diagrams, PNGs)

### Stimulus Types

#### **1. SVG (Scalable Vector Graphics)**

- **Use Cases**: Charts, graphs, diagrams, mathematical visualizations
- **Courses**: Statistics, Calculus, Physics, Biology, Microeconomics
- **Advantages**: Lightweight, scalable, fast to generate
- **Generation**: AI generates SVG markup directly

**Example:**

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">
  <rect x="50" y="200" width="50" height="100" fill="#4A90E2"/>
  <rect x="120" y="150" width="50" height="150" fill="#4A90E2"/>
  <!-- More bars for bar chart -->
</svg>
```

#### **2. Tables**

- **Use Cases**: Data tables, statistical data, experimental results
- **Courses**: Statistics, Science courses, Economics
- **Generation**: AI generates markdown table, converted to HTML
- **Format**: HTML table with proper styling

#### **3. Images (PNG/JPG)**

- **Use Cases**: Historical maps, primary source documents, photographs, diagrams
- **Courses**: History courses, some science courses
- **Generation Options**:
  - **Option A**: AI image generation (DALL-E, Stable Diffusion, Midjourney API)
  - **Option B**: Curated image database (public domain, properly licensed)
  - **Option C**: Hybrid (generate diagrams, source historical images)

**For History Courses:**

- Historical maps → AI generate or source from public domain
- Primary source documents → Source from archives (with licensing)
- Political cartoons → AI generate or curated database

#### **4. Passages (Text)**

- **Use Cases**: Reading passages for Lang/Lit courses
- **Courses**: AP English Language, AP English Literature
- **Generation Options**:
  - **Option A**: Public domain texts (Project Gutenberg)
  - **Option B**: AI-generated passages (matching AP complexity)
  - **Option C**: Hybrid (source real passages, generate questions)

### Stimulus Generation System

#### **Stimulus Factory Pattern**

```python
class StimulusFactory:
    def __init__(self, course_config):
        self.config = course_config
        self.generators = {}

        # Initialize generators based on config
        if "svg" in config["stimulus_config"]["supported_types"]:
            self.generators["svg"] = SVGGenerator()

        if "image" in config["stimulus_config"]["supported_types"]:
            use_ai = config["stimulus_config"]["generation_strategy"] == "external"
            self.generators["image"] = ImageGenerator(use_ai_generation=use_ai)

        if "passage" in config["stimulus_config"]["supported_types"]:
            use_public_domain = config["stimulus_config"]["generation_strategy"] == "inline"
            self.generators["passage"] = PassageGenerator(use_public_domain=use_public_domain)

    def generate_stimulus(self, stimulus_type, requirements):
        generator = self.get_generator(stimulus_type)
        return generator.generate(stimulus_type, requirements)
```

#### **SVG Generator**

- **Input**: Description, diagram type, data
- **Process**: AI generates SVG markup
- **Output**: Valid SVG string
- **Validation**: SVG syntax valid, viewBox correct

#### **Image Generator**

- **Input**: Description, image type, requirements
- **Process**:
  - If AI generation: Call image generation API
  - If database: Lookup in curated database
- **Output**: Image URL or base64 encoded image
- **Validation**: Image URL accessible or image data valid

#### **Passage Generator**

- **Input**: Genre, length, complexity
- **Process**:
  - If public domain: Source from database
  - If AI generation: Generate with AI
- **Output**: Formatted passage text
- **Validation**: Length correct, complexity appropriate

### Stimulus Decision Logic

```python
def should_include_stimulus(question_type, course_config, unit_payload):
    # Check course requirements
    if course_config["special_requirements"]["needs_passages"] and question_type == "frq":
        return True, "passage"

    if course_config["special_requirements"]["needs_historical_images"]:
        return True, "image"

    # Check unit-specific needs
    if "data_analysis" in unit_payload["unit"].lower():
        return True, "table"

    # Random chance based on config
    stimulus_ratio = course_config["generation_config"][question_type].get("stimulus_ratio", 0.3)
    if random.random() < stimulus_ratio:
        return True, course_config["stimulus_config"]["default_type"]

    return False, None
```

### Stimulus Validation

**SVG Validation:**

- Valid XML/SVG syntax
- Proper viewBox
- No external dependencies
- Escaped for JSON

**Image Validation:**

- URL accessible (if external)
- Image data valid (if base64)
- Proper format (PNG, JPG)
- Appropriate size

**Passage Validation:**

- Length within requirements
- Complexity appropriate
- Properly formatted
- No encoding issues

---

## ✅ Validation & Quality Assurance

### Multi-Stage Validation Pipeline

#### **Stage 1: Extraction Validation**

```
✓ PDF file exists
✓ Vision model successfully analyzes PDF structure
✓ All required sections found
✓ Page ranges valid
✓ Content extracted successfully
✓ Schema validation passed
✓ Learning objective IDs valid format
✓ Skill codes valid format
```

#### **Stage 2: Standardization Validation**

```
✓ Unified schema matches
✓ All required fields present
✓ Cross-references valid
✓ No data loss during normalization
```

#### **Stage 3: Generation Validation**

```
For MCQs:
✓ Schema validation
✓ Content validation
✓ Alignment validation (LOs, skills)
✓ Quality validation (difficulty, distractors)

For FRQs:
✓ Schema validation
✓ Part coherence validation
✓ Task verb validation
✓ Scoring guideline validation
```

#### **Stage 4: Output Validation**

```
✓ HTML renders correctly
✓ All questions display properly
✓ Stimuli render correctly
✓ Answer keys correct
✓ No broken links or images
```

### Quality Metrics

**Learning Objective Coverage:**

- All LOs in unit have questions
- Balanced coverage across LOs

**Skill Coverage:**

- All skills in unit are assessed
- Balanced skill distribution

**Difficulty Distribution:**

- Matches target distribution (e.g., 30% easy, 50% medium, 20% hard)

**Stimulus Coverage:**

- Appropriate percentage have stimuli
- Stimuli are necessary (not decorative)

**Question Quality:**

- Questions assess understanding (not just recall)
- Distractors reflect misconceptions
- Real-world context when appropriate

---

## 🤖 Models & Technology

### Models Used

#### **1. Gemini 2.5 Pro** (Primary)

- **Use Cases**:
  - MCQ generation
  - FRQ generation
  - Content normalization
  - Section extraction (after structure analysis)
- **Why**: High quality, good JSON output, cost-effective

#### **2. Gemini 2.5 Flash** (Secondary)

- **Use Cases**:
  - Fast content processing
  - Simple normalization tasks
- **Why**: Faster, cheaper for simple tasks

#### **3. Gemini 2.0 Pro Vision** (For PDF Analysis)

- **Use Cases**:
  - PDF structure analysis
  - Section detection in single PDFs
  - Image/diagram extraction from PDFs
- **Why**: Handles visual PDFs, finds sections automatically, understands layout

#### **4. Image Generation APIs** (For History Courses)

- **Options**:
  - DALL-E 3 (OpenAI)
  - Stable Diffusion API
  - Midjourney API
- **Use Cases**: Historical images, maps, diagrams for history courses
- **Decision**: Based on budget and quality requirements

### Technology Stack

**Python Libraries:**

- `pdfplumber`: Text extraction from PDFs (fallback)
- `google-genai`: Gemini API client
- `jsonschema`: Schema validation
- `jinja2`: HTML template rendering
- `pdf2image`: PDF to image conversion (for vision models)
- `Pillow`: Image processing

**File Formats:**

- **Input**: PDF (single CED documents)
- **Intermediate**: JSON (standardized content, question data)
- **Output**: HTML (question sets)

**Storage:**

- **Content**: `utils/content/{course_id}.json`
- **Configs**: `utils/config/{course_id}.json`, `utils/config/{course_id}_extraction.json`
- **Output**: `output/{course_id}/unit_{N}/mcqs/set_{M}.html`

---

## 🚀 Plan Moving Forward

### Phase 1: Foundation (Weeks 1-2)

**Goals:**

- Implement vision model-based extraction for single PDFs
- Create unified schema and normalization
- Build config system

**Tasks:**

1. ⏳ Create extraction config system for single PDFs
2. ⏳ Implement PDF structure analysis with vision model
3. ⏳ Implement section-by-section extraction
4. ⏳ Build schema validator and normalizer
5. ⏳ Test with AP Microeconomics (single PDF)

**Deliverables:**

- Extraction system working for single PDFs
- Standardized JSON for at least one course
- Config system functional

### Phase 2: Core Generation (Weeks 3-4)

**Goals:**

- Implement MCQ and FRQ generators
- Build validation pipeline
- Create stimulus generators

**Tasks:**

1. ⏳ Refactor MCQ generator (use shared payload builder)
2. ⏳ Refactor FRQ generator (use shared payload builder)
3. ⏳ Implement SVG generator
4. ⏳ Implement table formatter
5. ⏳ Build validation pipeline
6. ⏳ Test with extracted course content

**Deliverables:**

- MCQ and FRQ generators working
- Validation pipeline functional
- SVG and table stimuli working

### Phase 3: Course Expansion (Weeks 5-6)

**Goals:**

- Add support for image generation
- Add support for passage generation
- Test with multiple courses

**Tasks:**

1. ⏳ Implement image generator (AI or database)
2. ⏳ Implement passage generator (public domain or AI)
3. ⏳ Test with AP Microeconomics (single PDF)
4. ⏳ Test with AP English Literature (passages)
5. ⏳ Test with AP US History (images)

**Deliverables:**

- Image generation working
- Passage generation working
- At least 3 courses working end-to-end

### Phase 4: Optimization & Scaling (Weeks 7-8)

**Goals:**

- Optimize performance
- Add caching
- Scale to more courses

**Tasks:**

1. ⏳ Implement caching system
2. ⏳ Optimize batch sizes
3. ⏳ Add quality metrics
4. ⏳ Document system
5. ⏳ Scale to 5-10 courses

**Deliverables:**

- Caching system working
- Performance optimized
- 5-10 courses generating questions
- Complete documentation

---

## 🎯 Most Important Parts to Capture

### Critical (Must Have)

1. **Learning Objectives** → Questions MUST align to these
2. **Essential Knowledge** → Ensures content accuracy
3. **Skills** → Questions must assess these skills
4. **Task Verbs** → FRQs must use correct verbs
5. **Exam Sections** → Questions must match exam format

### Important (Should Have)

6. **Big Ideas** → Questions should connect to concepts
7. **Unit Context** → Developing understanding, preparing for exam
8. **Topics** → Organize questions by topic

### Nice to Have (Future Enhancements)

9. **Common Misconceptions** → Better distractors
10. **Prerequisites** → Question progression
11. **Difficulty Indicators** → Better difficulty distribution

---

## 📐 How to Structure

### Directory Structure

```
BridgeUp/
├── ap_specs/                    # Input CED PDFs (single PDFs)
│   ├── ap_microeconomics.pdf
│   ├── ap_english_literature_and_composition.pdf
│   ├── ap_physics_1.pdf
│   └── ...
├── utils/
│   ├── config/                  # Configuration files
│   │   ├── ap_microeconomics.json          # Course configs
│   │   ├── ap_microeconomics_extraction.json  # Extraction configs
│   │   └── ...
├── utils/
│   ├── content/                 # Extracted standardized JSON
│   │   ├── ap_microeconomics.json
│   │   └── ...
│   ├── prompts/                 # Prompt templates
│   │   ├── mcq_prompt.txt
│   │   └── frq_prompt.txt
│   ├── schemas/                 # JSON schemas
│   │   ├── mcq.schema.json
│   │   └── frq.schema.json
│   ├── templates/               # HTML templates
│   │   ├── mcq.html
│   │   └── frq.html
│   ├── payload_builder.py       # Shared payload builder
│   ├── stimulus_generators.py   # Stimulus generation
│   └── config_loader.py         # Config loading
├── output/                      # Generated question sets
│   ├── ap_microeconomics/
│   │   └── unit_1/
│   │       ├── mcqs/
│   │       └── frqs/
│   └── ...
├── pdf_to_json.py              # Extraction script
├── mcq_compiler.py             # MCQ generation
├── frq_compiler.py             # FRQ generation
└── COMPLETE_GUIDE.md          # This file
```

### File Naming Conventions

- **Extraction configs**: `{course_id}_extraction.json`
- **Course configs**: `{course_id}.json`
- **Content JSON**: `{course_id}.json`
- **Output HTML**: `set_{N}.html`

---

## 📊 Summary

### The System

A config-driven pipeline that extracts content from single AP CED PDFs using vision models and generates high-quality practice questions (MCQs and FRQs) that align to learning objectives, assess correct skills, and match exam style.

### Key Innovations

1. **Vision model extraction** → Automatically finds sections in single PDFs
2. **Config-driven generation** → Handles course variance
3. **Unified schema** → Consistent structure
4. **Multi-stage validation** → Ensures quality
5. **Stimulus abstraction** → Handles all stimulus types

### What Makes Questions Good

- ✅ Aligned to learning objectives
- ✅ Assess correct skills
- ✅ Match exam style
- ✅ Have appropriate stimulus
- ✅ Use correct format
- ✅ Reflect common misconceptions (distractors)

### Next Steps

1. Implement vision model-based extraction for single PDFs
2. Refactor generators to use shared code
3. Add image and passage generation
4. Test with multiple courses
5. Scale to all AP courses

---

This system provides a complete, scalable solution for generating AP practice questions from single CED PDFs while handling the variance across different courses and PDF layouts.

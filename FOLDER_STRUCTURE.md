# Complete Folder Structure & Schema Files

## ✅ Verified Folder Structure

```
BridgeUp/
├── courses/                          ✅ Raw CED PDFs
│   ├── ap_statistics.pdf
│   ├── ap_microeconomics.pdf
│   └── ... (all AP course PDFs)
│
├── utils/
│   ├── config/                      ✅ All configuration files
│   │   ├── ap_statistics.json                    (course config)
│   │   ├── ap_statistics_extraction.json         (extraction config - auto-generated)
│   │   └── {course_id}.json                     (course configs)
│   │
│   ├── content/                     ✅ Extracted standardized JSON
│   │   ├── ap_statistics.json
│   │   └── {course_id}.json
│   │
│   ├── prompts/                     ✅ LLM prompt templates
│   │   ├── mcq_prompt.txt
│   │   └── frq_prompt.txt
│   │
│   ├── schemas/                     ✅ JSON schemas for validation
│   │   ├── content_schema.json      ✅ Validates extracted CED content
│   │   ├── mcq.schema.json          ✅ Validates MCQ generation output
│   │   └── frq.schema.json          ✅ Validates FRQ generation output
│   │
│   └── templates/                   ✅ HTML templates
│       ├── mcq.html
│       └── frq.html
│
├── output/                           ✅ Generated question sets
│   └── {course_id}/
│       ├── mcq/
│       │   ├── set_1.html
│       │   └── set_2.html
│       └── frq/
│           ├── set_1.html
│           └── set_2.html
│
└── pdf_to_json.py                   ✅ Phase 1 extraction script
```

## ✅ Schema Files Status

### 1. **content_schema.json** ✅ CREATED

- **Location**: `utils/schemas/content_schema.json`
- **Purpose**: Validates extracted CED content from PDFs
- **Validates**:
  - ✅ Course metadata (course_id, name, extraction_method, extraction_date)
  - ✅ Skills (skill_name, skill_description, subskills with codes)
  - ✅ Big ideas (id, name, description)
  - ✅ Units (name, developing_understanding, building_practices, preparing_for_exam)
  - ✅ Topics (name, big_ideas, suggested_subskill_codes)
  - ✅ Learning objectives (id, description, essential_knowledge)
  - ✅ Exam sections (section, question_type, timing, weighting, descriptions)
  - ✅ Task verbs (verb, description) - optional
- **Status**: ✅ Created, valid JSON, ready for validation

### 2. **mcq.schema.json** ✅ EXISTS

- **Location**: `utils/schemas/mcq.schema.json`
- **Purpose**: Validates MCQ question generation output
- **Validates**: Questions, choices, correct answer, explanations, alignment, stimulus
- **Status**: ✅ Exists and validated

### 3. **frq.schema.json** ✅ EXISTS

- **Location**: `utils/schemas/frq.schema.json`
- **Purpose**: Validates FRQ question generation output
- **Validates**: FRQ context, parts, scoring guidelines, alignment, stimulus
- **Status**: ✅ Exists and validated

## 🔍 Schema Validation Flow

### Phase 1: Content Extraction

```
PDF → Vision Model → Raw JSON
                    ↓
            content_schema.json (validation)
                    ↓
            utils/content/{course_id}.json (validated output)
```

### Phase 2: MCQ Generation

```
Unit Payload → LLM → MCQ JSON
                    ↓
            mcq.schema.json (validation)
                    ↓
            Validated MCQ questions
```

### Phase 3: FRQ Generation

```
Unit Payload → LLM → FRQ JSON
                    ↓
            frq.schema.json (validation)
                    ↓
            Validated FRQ questions
```

## ✅ All Required Files Present

| File                | Location         | Status     | Purpose                         |
| ------------------- | ---------------- | ---------- | ------------------------------- |
| content_schema.json | `utils/schemas/` | ✅ Created | Validates extracted CED content |
| mcq.schema.json     | `utils/schemas/` | ✅ Exists  | Validates MCQ generation        |
| frq.schema.json     | `utils/schemas/` | ✅ Exists  | Validates FRQ generation        |

## 📋 Directory Checklist

- ✅ `courses/` - Raw CED PDFs
- ✅ `utils/config/` - Course & extraction configs
- ✅ `utils/content/` - Extracted standardized JSON
- ✅ `utils/prompts/` - LLM prompt templates
- ✅ `utils/schemas/` - JSON schemas (all 3 present)
- ✅ `utils/templates/` - HTML templates
- ✅ `output/` - Generated question sets

## 🎯 Ready for Use

All folder structure is in place and all schema files are ready for validation:

1. ✅ **content_schema.json** - Will validate all new extractions
2. ✅ **mcq.schema.json** - Ready for MCQ generation validation
3. ✅ **frq.schema.json** - Ready for FRQ generation validation

The system is ready to:

- Extract content from PDFs (validates with content_schema.json)
- Generate MCQs (validates with mcq.schema.json)
- Generate FRQs (validates with frq.schema.json)

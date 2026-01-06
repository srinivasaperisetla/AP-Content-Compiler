# Folder Structure & Schema Verification

## ✅ Complete Directory Structure

```
BridgeUp/
├── courses/                          ✅ Raw CED PDFs
│   ├── ap_statistics.pdf
│   ├── ap_microeconomics.pdf
│   └── ...
│
├── utils/
│   ├── config/                      ✅ Course & extraction configs
│   │   ├── ap_statistics.json
│   │   ├── {course_id}.json
│   │   └── {course_id}_extraction.json (auto-generated)
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
│   │   ├── content_schema.json      ✅ For extracted content validation
│   │   ├── mcq.schema.json          ✅ For MCQ generation validation
│   │   └── frq.schema.json          ✅ For FRQ generation validation
│   │
│   └── templates/                    ✅ HTML templates
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

### 1. **content_schema.json** ✅

- **Location**: `utils/schemas/content_schema.json`
- **Purpose**: Validates extracted CED content
- **Validates**:
  - Course metadata
  - Skills (with subskills)
  - Big ideas
  - Units (with topics, learning objectives, essential knowledge)
  - Exam sections
  - Task verbs
- **Status**: ✅ Created and validated

### 2. **mcq.schema.json** ✅

- **Location**: `utils/schemas/mcq.schema.json`
- **Purpose**: Validates MCQ question generation output
- **Validates**: Questions, choices, correct answer, explanations, alignment
- **Status**: ✅ Exists and validated

### 3. **frq.schema.json** ✅

- **Location**: `utils/schemas/frq.schema.json`
- **Purpose**: Validates FRQ question generation output
- **Validates**: FRQ context, parts, scoring guidelines, alignment
- **Status**: ✅ Exists and validated

## 📋 Schema Validation Flow

### Phase 1: Content Extraction

```
PDF → Vision Model → Raw JSON → content_schema.json → Validated JSON
                                                      ↓
                                              utils/content/{course_id}.json
```

### Phase 2: Question Generation

```
Unit Payload → LLM → MCQ JSON → mcq.schema.json → Validated MCQ
Unit Payload → LLM → FRQ JSON → frq.schema.json → Validated FRQ
```

## 🔍 Verification Commands

### Check directory structure:

```bash
python3 -c "
import os
dirs = ['courses', 'utils/config', 'utils/content', 'utils/prompts',
        'utils/schemas', 'utils/templates', 'output']
for d in dirs:
    print(f'{'✅' if os.path.exists(d) else '❌'} {d}/')
"
```

### Validate schema files:

```bash
python3 -c "
import json
schemas = ['utils/schemas/content_schema.json',
           'utils/schemas/mcq.schema.json',
           'utils/schemas/frq.schema.json']
for s in schemas:
    with open(s) as f:
        json.load(f)
    print(f'✅ {s} is valid JSON')
"
```

### Test schema validation:

```bash
python3 -c "
from jsonschema import validate
import json

# Load schema
with open('utils/schemas/content_schema.json') as f:
    schema = json.load(f)

# Load sample content
with open('utils/content/ap_statistics.json') as f:
    content = json.load(f)

# Validate
validate(instance=content, schema=schema)
print('✅ Content validates against schema!')
"
```

## ✅ All Required Files Present

- ✅ `utils/schemas/content_schema.json` - Content validation
- ✅ `utils/schemas/mcq.schema.json` - MCQ validation
- ✅ `utils/schemas/frq.schema.json` - FRQ validation
- ✅ All directories created
- ✅ All schemas are valid JSON
- ✅ Schema validation ready for use

## 🎯 Next Steps

1. **Phase 1**: Run `pdf_to_json.py` to extract content (uses `content_schema.json`)
2. **Phase 2**: Generate MCQs (validates with `mcq.schema.json`)
3. **Phase 3**: Generate FRQs (validates with `frq.schema.json`)

All schema validation is in place and ready to use!

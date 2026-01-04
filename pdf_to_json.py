import os
import json
import pdfplumber
from dotenv import load_dotenv
from google import genai

AP_COURSES = [
  "ap_statistics"
]

for course in AP_COURSES:

  BASE_DIR = f"ap_specs/{course}"
  OUTPUT_PATH = f"utils/content/{course}.json"
  UNITS_DIR = f"ap_specs/{course}/units"
  TEMPLATE_PATH = "ap_specs/1template.json"

  PDF_FILES = {
    "skills": f"skills_{course}.pdf",
    "big_ideas": f"big_ideas_{course}.pdf",
    "exam_sections": f"exam_sections_{course}.pdf",
    "task_verbs" : f"task_verbs_{course}.pdf"
  }

  load_dotenv()
  API_KEY = os.getenv("GEMINI_API_KEY")
  assert API_KEY, "Missing GEMINI_API_KEY"
  client = genai.Client(api_key=API_KEY)

  def extract_pdf_text(path):
    text = []
    with pdfplumber.open(path) as pdf:
      for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
          text.append(page_text)
    return "\n\n".join(text)

  def load_template():
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
      return json.load(f)

  print("📄 Extracting PDFs...")
  static_texts = {
    key: extract_pdf_text(os.path.join(BASE_DIR, fname))
    for key, fname in PDF_FILES.items()
  }

  print("📄 Extracting unit PDFs...")
  unit_files = sorted(f for f in os.listdir(UNITS_DIR) if f.endswith(".pdf"))

  unit_texts = [
    {
      "unit_file": f,
      "text": extract_pdf_text(os.path.join(UNITS_DIR, f))
    }
    for f in unit_files
  ]

  template = load_template()

  SYSTEM_PROMPT = f"""
  You are an AP Course Framework parser.

  Your job:
  - Read authoritative course framework text
  - Populate a JSON object EXACTLY matching this template
  - Do NOT invent information
  - Use ONLY the provided text and copy and paste
  - Preserve wording faithfully where appropriate

  JSON TEMPLATE:
  {json.dumps(template, indent=2)}

  Rules:
  - Output VALID JSON only
  - No markdown
  - No commentary
  """

  contents = []

  contents.append({
    "parts": [{
      "text": SYSTEM_PROMPT
    }]
  })

  for section, text in static_texts.items():
    contents.append({
      "parts": [{
          "text": f"""
          The following text corresponds to the **{section.upper()}** section of the course framework.

          Populate ONLY the relevant fields.

          TEXT:
          {text}
          """
      }]
    })


  for unit in unit_texts:
    contents.append({
      "parts": [{
        "text": f"""
        The following text corresponds to a SINGLE COURSE UNIT.

        Extract:
        - Unit name
        - Developing Understanding
        - Building Practices
        - Preparing for Exam
        - Topics
        - Learning Objectives
        - Essential Knowledge

        Append this unit as a NEW entry in the "units" array.

        SOURCE FILE: {unit['unit_file']}

        TEXT:
        {unit['text']}
        """
      }]
    })

  contents.append({
    "parts": [{
      "text": "Now return the COMPLETE populated JSON object."
    }]
  })

  print("🤖 Generating JSON with Gemini...")
  response = client.models.generate_content(
      model="gemini-2.5-flash",
      contents=contents,
      config={
        "response_mime_type": "application/json"
      }
  )

  try:
    parsed = json.loads(response.text)
  except json.JSONDecodeError as e:
    raise RuntimeError("Gemini did not return valid JSON") from e

  with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(parsed, f, indent=2, ensure_ascii=False)

  print(f"✅ JSON successfully generated → {OUTPUT_PATH}")
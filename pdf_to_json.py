import os
import json
import pdfplumber
from dotenv import load_dotenv
from google import genai
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()

AP_COURSES = ["ap_statistics"]

# -----------------------
# INIT
# -----------------------

print("🔧 Initializing environment...")
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
assert API_KEY, "Missing GEMINI_API_KEY"
client = genai.Client(api_key=API_KEY)
print("✅ Gemini client initialized")

# -----------------------
# HELPERS
# -----------------------

def extract_pdf_text(path):
	print(f"    📄 Extracting text from PDF: {path}")
	text = []
	with pdfplumber.open(path) as pdf:
		for i, page in enumerate(pdf.pages, start=1):
			page_text = page.extract_text()
			if page_text:
				text.append(page_text)
			else:
				print(f"      ⚠️ Page {i} had no extractable text")
	return "\n\n".join(text)

def load_template(path):
	print(f"📄 Loading JSON template: {path}")
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)

def generate_json_with_retry(model, contents, max_retries=3):
	for attempt in range(1, max_retries + 1):
		print(f"    🤖 Gemini attempt {attempt}/{max_retries}")
		response = client.models.generate_content(
			model=model,
			contents=contents,
			config={"response_mime_type": "application/json"}
		)

		raw = response.text.strip()

		try:
			parsed = json.loads(raw)
			print("    ✅ Valid JSON received")
			return parsed
		except json.JSONDecodeError:
			print("    ❌ Invalid JSON returned by Gemini")

			with open(f"debug_gemini_output_attempt_{attempt}.txt", "w", encoding="utf-8") as f:
				f.write(raw)

			if attempt == max_retries:
				raise RuntimeError("Gemini failed to return valid JSON after retries")

# -----------------------
# MAIN PIPELINE
# -----------------------

for course in AP_COURSES:
	print(f"\n📘 Starting course processing: {course}")

	BASE_DIR = f"ap_specs/{course}"
	UNITS_DIR = f"{BASE_DIR}/units"
	TEMPLATE_PATH = "ap_specs/1template.json"
	OUTPUT_PATH = f"utils/content/{course}.json"

	PDF_FILES = {
		"skills": f"skills_{course}.pdf",
		"big_ideas": f"big_ideas_{course}.pdf",
		"exam_sections": f"exam_sections_{course}.pdf",
		"task_verbs": f"task_verbs_{course}.pdf"
	}

	result = load_template(TEMPLATE_PATH)
	result["name"] = course.replace("_", " ").title()

	# -----------------------
	# STATIC SECTIONS
	# -----------------------

	print("\n📚 Processing static course sections...")

	for section, filename in PDF_FILES.items():
		print(f"\n🔹 Parsing section: {section}")
		pdf_path = os.path.join(BASE_DIR, filename)

		text = extract_pdf_text(pdf_path)

		contents = [{
			"parts": [{
				"text": f"""
You are parsing the **{section}** section of an AP course framework.

Populate ONLY the `{section}` field of the JSON.

TEXT:
{text}
"""
			}]
		}]

		parsed_section = generate_json_with_retry(
			model="gemini-2.5-flash",
			contents=contents,
			max_retries=3
		)

		result[section] = parsed_section[section]
		print(f"✅ Finished parsing section: {section}")

	# -----------------------
	# UNITS
	# -----------------------

	print("\n📘 Processing course units...")
	result["units"] = []

	unit_files = sorted(f for f in os.listdir(UNITS_DIR) if f.endswith(".pdf"))

	for i, unit_file in enumerate(unit_files, start=1):
		print(f"\n🧩 Parsing Unit {i}: {unit_file}")

		unit_path = os.path.join(UNITS_DIR, unit_file)
		unit_text = extract_pdf_text(unit_path)

		contents = [{
			"parts": [{
				"text": f"""
You are parsing ONE AP course unit.

Return a SINGLE unit object matching the template exactly.

TEXT:
{unit_text}
"""
			}]
		}]

		unit_json = generate_json_with_retry(
			model="gemini-2.5-flash",
			contents=contents,
			max_retries=3
		)

		result["units"].append(unit_json)
		print(f"✅ Unit {i} parsed and appended")

	# -----------------------
	# SAVE OUTPUT
	# -----------------------

	print("\n💾 Writing final JSON output...")
	with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
		json.dump(result, f, indent=2, ensure_ascii=False)

	print(f"🎉 Course JSON successfully generated → {OUTPUT_PATH}")

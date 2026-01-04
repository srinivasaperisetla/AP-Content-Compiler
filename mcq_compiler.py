import os
import json
from dotenv import load_dotenv
from google import genai
from jsonschema import validate
from jinja2 import Template

# -----------------------
# CONFIG
# -----------------------

AP_COURSES = [
	"ap_statistics"
]

NUM_SETS_PER_UNIT = 1  # Later: 20 sets
QUESTIONS_PER_SET = 25  # one "set" = 25 questions

BASE_SPEC_DIR = "utils"
OUTPUT_BASE_DIR = "output"

MCQ_SCHEMA_PATH = "utils/schemas/mcq.schema.json"
MCQ_HTML_TEMPLATE_PATH = "utils/templates/mcq.html"
MCQ_PROMPT_PATH = "utils/prompts/mcq_prompt.txt"

MODEL = "gemini-2.5-pro"

# -----------------------
# INIT
# -----------------------

load_dotenv()
assert os.getenv("GEMINI_API_KEY"), "Missing GEMINI_API_KEY"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# -----------------------
# HELPERS
# -----------------------

def markdown_table_to_html(md: str) -> str:
	lines = [line.strip() for line in md.splitlines() if line.strip()]
	if len(lines) < 2:
		return f"<pre>{md}</pre>"

	headers = [h.strip() for h in lines[0].split("|")]
	rows = [
		[cell.strip() for cell in line.split("|")]
		for line in lines[2:]
	]

	html = "<table class='data-table'><thead><tr>"
	for h in headers:
		html += f"<th>{h}</th>"
	html += "</tr></thead><tbody>"

	for row in rows:
		html += "<tr>"
		for cell in row:
			html += f"<td>{cell}</td>"
		html += "</tr>"

	html += "</tbody></table>"
	return html

def strip_choice_labels(choice: str) -> str:
	return choice.lstrip("ABCD. ").strip()

def load_json(path: str):
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)

def load_text(path: str) -> str:
	with open(path, "r", encoding="utf-8") as f:
		return f.read()

def ensure_dir(path: str):
	os.makedirs(path, exist_ok=True)

def build_unit_payload(course_name: str, unit_index: int, unit: dict, course_spec: dict) -> dict:
	learning_objectives = []
	for topic in unit.get("topics", []):
		for lo in topic.get("learning_objectives", []):
			learning_objectives.append({
				"id": lo.get("id", ""),
				"description": lo.get("description", ""),
				"essential_knowledge": lo.get("essential_knowledge", []) or []
			})

	skill_codes = sorted({
		code
		for topic in unit.get("topics", [])
		for code in topic.get("suggested_subskill_codes", []) or []
	})

	big_ideas = sorted({
		bi
		for topic in unit.get("topics", [])
		for bi in topic.get("big_ideas", []) or []
	})

	skill_definitions = {}
	for skill in course_spec.get("skills", []):
		for sub in skill.get("subskills", []):
			code = sub.get("subskill_name")
			if code in skill_codes:
				skill_definitions[code] = {
					"category": skill.get("skill_name"),
					"description": sub.get("subskill_description")
				}

	big_idea_definitions = {}
	for bi in course_spec.get("big_ideas", []):
		if bi.get("id") in big_ideas:
			big_idea_definitions[bi["id"]] = {
				"name": bi.get("name"),
				"description": bi.get("description")
			}

	mcq_exam_context = None
	for section in course_spec.get("exam_sections", []):
		if section.get("section") == "I":
			mcq_exam_context = {
				"question_type": section.get("question_type"),
				"exam_weighting": section.get("exam_weighting"),
				"timing": section.get("timing"),
				"descriptions": section.get("descriptions", [])
			}
			break

	return {
		"course": course_name,
		"unit": f"Unit {unit_index + 1}: {unit.get('name','')}",
		"developing_understanding": unit.get("developing_understanding", ""),
		"building_practices": unit.get("building_practices", ""),
		"preparing_for_exam": unit.get("preparing_for_exam", ""),
		"learning_objectives": learning_objectives,
		"skill_codes": skill_codes,
		"skill_definitions": skill_definitions,
		"big_ideas": big_ideas,
		"big_idea_definitions": big_idea_definitions,
		"exam_section_context": mcq_exam_context
	}


# -----------------------
# MCQ GENERATION (BATCHED)
# -----------------------

def generate_valid_mcqs(prompt_template: Template, unit_payload: dict, total_questions: int, batch_size: int = 5, max_batches: int = 20):
	assert batch_size > 0 and batch_size <= total_questions

	all_questions = []
	seen_ids = set()
	batches_attempted = 0

	print(f"    🔁 Target MCQs: {total_questions} (batch size = {batch_size})")

	while len(all_questions) < total_questions and batches_attempted < max_batches:
		batches_attempted += 1
		remaining = total_questions - len(all_questions)
		n = min(batch_size, remaining)

		print(f"      ▶ Batch {batches_attempted}: requesting {n} MCQ(s) ({len(all_questions)}/{total_questions} collected)")

		prompt = prompt_template.render(
			num_questions=n,
			topic_payload=json.dumps(unit_payload, indent=2),
			schema=json.dumps(MCQ_SCHEMA, indent=2)
		)

		response = client.models.generate_content(
			model=MODEL,
			contents=prompt,
			config={"response_mime_type": "application/json"}
		)

		raw = response.text.strip().replace("\u0000", "").replace("\r", "")

		try:
			parsed = json.loads(raw)
			validate(instance=parsed, schema=MCQ_SCHEMA)
		except Exception:
			print(f"      ⚠️ Batch {batches_attempted}: invalid JSON/schema — retrying")
			continue

		added_this_batch = 0

		for q in parsed.get("questions", []):
			qid = q.get("id")
			if not qid or qid in seen_ids:
				print(f"        ↪ Skipped duplicate or missing MCQ id")
				continue

			try:
				validate(instance=q, schema=QUESTION_SCHEMA)
			except Exception:
				print(f"        ⚠️ MCQ {qid}: failed item-level validation")
				continue

			all_questions.append(q)
			seen_ids.add(qid)
			added_this_batch += 1

			if len(all_questions) >= total_questions:
				break

		print(f"      ✅ Batch {batches_attempted}: added {added_this_batch} MCQ(s)")

	if len(all_questions) < total_questions:
		raise RuntimeError(
			f"Failed to generate {total_questions} MCQs "
			f"(got {len(all_questions)} after {batches_attempted} batches)"
		)

	print(f"    🎯 Successfully generated {len(all_questions)} MCQs")
	return {"questions": all_questions[:total_questions]}

# -----------------------
# LOAD STATIC ASSETS
# -----------------------

MCQ_SCHEMA = load_json(MCQ_SCHEMA_PATH)
QUESTION_SCHEMA = MCQ_SCHEMA["properties"]["questions"]["items"]

MCQ_HTML_TEMPLATE = Template(load_text(MCQ_HTML_TEMPLATE_PATH))
MCQ_PROMPT_TEMPLATE = Template(load_text(MCQ_PROMPT_PATH))

# -----------------------
# MAIN LOOP
# -----------------------

for course in AP_COURSES:
	print(f"\n📘 Processing course: {course}")

	course_spec_path = f"{BASE_SPEC_DIR}/content/{course}.json"
	course_spec = load_json(course_spec_path)

	course_name = course_spec.get("name", course)
	units = course_spec.get("units", [])

	for unit_index, unit in enumerate(units):
		unit_title = unit.get("name", f"Unit {unit_index + 1}")
		print(f"  📄 Generating MCQ SETS for Unit {unit_index + 1}: {unit_title}")

		unit_payload = build_unit_payload(course_name, unit_index, unit, course_spec)

		output_dir = f"{OUTPUT_BASE_DIR}/{course}/unit_{unit_index + 1}/mcqs"
		ensure_dir(output_dir)

		for set_index in range(NUM_SETS_PER_UNIT):
			print(f"    🧩 Starting MCQ Set {set_index + 1}")

			mcq_json = generate_valid_mcqs(
				prompt_template=MCQ_PROMPT_TEMPLATE,
				unit_payload=unit_payload,
				total_questions=QUESTIONS_PER_SET,
				batch_size=5
			)

			for q in mcq_json["questions"]:
				q["choices"] = [strip_choice_labels(c) for c in q["choices"]]

			for q in mcq_json["questions"]:
				stim = q.get("stimulus")
				if stim and stim["type"] == "table":
					stim["content"] = markdown_table_to_html(stim["content"])

			html_output = MCQ_HTML_TEMPLATE.render(
				course=course_name,
				unit=f"Unit {unit_index + 1}: {unit_title}",
				questions=mcq_json["questions"]
			)

			output_path = f"{output_dir}/set_{set_index + 1}.html"
			with open(output_path, "w", encoding="utf-8") as f:
				f.write(html_output)

			print(f"    ✅ Saved → {output_path}")

print("\n🎉 All MCQ sets generated successfully.")

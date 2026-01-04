import os
import json
from dotenv import load_dotenv
from google import genai
from jsonschema import validate
from jinja2 import Template

# -----------------------
# CONFIG
# -----------------------

AP_COURSES = ["ap_statistics"]

NUM_SETS_PER_UNIT = 1  # Later 10 sets
FRQS_PER_SET = 25

BASE_SPEC_DIR = "utils"
OUTPUT_BASE_DIR = "output"

FRQ_SCHEMA_PATH = "utils/schemas/frq.schema.json"
FRQ_HTML_TEMPLATE_PATH = "utils/templates/frq.html"
FRQ_PROMPT_PATH = "utils/prompts/frq_prompt.txt"

MODEL = "gemini-2.5-pro"

# -----------------------
# INIT
# -----------------------

load_dotenv()
assert os.getenv("GEMINI_API_KEY"), "Missing GEMINI_API_KEY"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# -----------------------
# SHARED HELPERS
# -----------------------

def load_json(path):
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)

def load_text(path):
	with open(path, "r", encoding="utf-8") as f:
		return f.read()

def ensure_dir(path):
	os.makedirs(path, exist_ok=True)

def build_unit_payload(course_name, unit_index, unit, course_spec):
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

	frq_exam_context = None
	for section in course_spec.get("exam_sections", []):
		if section.get("section") == "II":
			frq_exam_context = {
				"question_type": section.get("question_type"),
				"exam_weighting": section.get("exam_weighting"),
				"timing": section.get("timing"),
				"descriptions": section.get("descriptions", [])
			}
			break

	task_verbs = []
	for verb in course_spec.get("task_verbs", []):
		task_verbs.append({
			"verb": verb.get("verb"),
			"description": verb.get("description")
		})

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
		"exam_section_context": frq_exam_context,
		"task_verbs": task_verbs
	}


# -----------------------
# FRQ GENERATION (BATCHED)
# -----------------------

def generate_valid_frqs(prompt_template, unit_payload, total_frqs, batch_size=1, max_batches=50):
	all_frqs = []
	seen_ids = set()
	batches = 0

	print(f"    🔁 Target FRQs: {total_frqs} (batch size = {batch_size})")

	while len(all_frqs) < total_frqs and batches < max_batches:
		batches += 1
		remaining = total_frqs - len(all_frqs)
		n = min(batch_size, remaining)

		print(f"      ▶ Batch {batches}: requesting {n} FRQ(s) ({len(all_frqs)}/{total_frqs} collected)")

		prompt = prompt_template.render(
			num_frqs=n,
			start_index=len(all_frqs) + 1,
			topic_payload=json.dumps(unit_payload, indent=2),
			schema=json.dumps(FRQ_SCHEMA, indent=2)
		)

		response = client.models.generate_content(
			model=MODEL,
			contents=prompt,
			config={"response_mime_type": "application/json"}
		)

		raw = response.text.strip().replace("\u0000", "").replace("\r", "")

		try:
			parsed = json.loads(raw)
			validate(parsed, FRQ_SCHEMA)
		except Exception:
			print(f"      ⚠️ Batch {batches}: invalid FRQ JSON/schema — retrying")
			continue

		added_this_batch = 0

		for frq in parsed.get("frqs", []):
			fid = frq.get("id")
			# if not fid or fid in seen_ids:
			# 	print(f"        ↪ Skipped duplicate or missing FRQ id")
			# 	continue

			try:
				validate(frq, FRQ_ITEM_SCHEMA)
			except Exception:
				print(f"        ⚠️ FRQ {fid}: failed item-level validation")
				continue

			all_frqs.append(frq)
			seen_ids.add(fid)
			added_this_batch += 1

			if len(all_frqs) >= total_frqs:
				break

		print(f"      ✅ Batch {batches}: added {added_this_batch} FRQ(s)")

	if len(all_frqs) < total_frqs:
		raise RuntimeError(
			f"Failed to generate {total_frqs} FRQs "
			f"(got {len(all_frqs)} after {batches} batches)"
		)

	print(f"    🎯 Successfully generated {len(all_frqs)} FRQs")
	return {"frqs": all_frqs[:total_frqs]}

# -----------------------
# LOAD STATIC ASSETS
# -----------------------

FRQ_SCHEMA = load_json(FRQ_SCHEMA_PATH)
FRQ_ITEM_SCHEMA = FRQ_SCHEMA["properties"]["frqs"]["items"]

FRQ_HTML_TEMPLATE = Template(load_text(FRQ_HTML_TEMPLATE_PATH))
FRQ_PROMPT_TEMPLATE = Template(load_text(FRQ_PROMPT_PATH))

# -----------------------
# MAIN LOOP
# -----------------------

for course in AP_COURSES:
	print(f"\n📘 Processing course: {course}")

	course_spec = load_json(f"{BASE_SPEC_DIR}/content/{course}.json")
	course_name = course_spec.get("name", course)

	for unit_index, unit in enumerate(course_spec.get("units", [])):
		unit_title = unit.get("name", f"Unit {unit_index + 1}")
		print(f"  📄 Generating FRQ SETS for Unit {unit_index + 1}: {unit_title}")

		unit_payload = build_unit_payload(course_name, unit_index, unit, course_spec)

		output_dir = f"{OUTPUT_BASE_DIR}/{course}/unit_{unit_index + 1}/frqs"
		ensure_dir(output_dir)

		for set_index in range(NUM_SETS_PER_UNIT):
			print(f"    🧩 Starting FRQ Set {set_index + 1}")

			frq_json = generate_valid_frqs(
				FRQ_PROMPT_TEMPLATE,
				unit_payload,
				FRQS_PER_SET,
				batch_size=5
			)

			html = FRQ_HTML_TEMPLATE.render(
				course=course_name,
				unit=f"Unit {unit_index + 1}: {unit_title}",
				set_number=set_index + 1,
				frqs=frq_json["frqs"]
			)

			path = f"{output_dir}/set_{set_index + 1}.html"
			with open(path, "w", encoding="utf-8") as f:
				f.write(html)

			print(f"    ✅ Saved → {path}")

print("\n🎉 All FRQ sets generated successfully.")

import os
import json
from dotenv import load_dotenv
from google import genai
from jsonschema import validate, ValidationError
from jinja2 import Template
import certifi
import time
import re

os.environ["SSL_CERT_FILE"] = certifi.where()

# -----------------------
# CONFIG
# -----------------------

AP_COURSES = ["ap_statistics"]

NUM_SETS_PER_UNIT = 20
QUESTIONS_PER_SET = 25

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
	rows = [[cell.strip() for cell in line.split("|")] for line in lines[2:]]

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

def mcq_set_exists(course, unit_index, set_index):
	path = f"{OUTPUT_BASE_DIR}/{course}/unit_{unit_index + 1}/mcqs/set_{set_index + 1}.html"
	return os.path.exists(path)

def validate_unit_payload(unit_payload: dict):
	"""
	Validate that unit payload has sufficient metadata for MCQ generation.
	Returns (is_valid, error_message)
	"""
	has_learning_objectives = len(unit_payload.get('learning_objectives', [])) > 0
	has_skill_codes = len(unit_payload.get('skill_codes', [])) > 0
	
	if not has_learning_objectives:
		return False, "No learning objectives found in unit payload"
	
	if not has_skill_codes:
		return False, "No skill codes found in unit payload"
	
	# Validate that learning objectives have IDs
	for lo in unit_payload.get('learning_objectives', []):
		if not lo.get('id'):
			return False, f"Learning objective missing ID: {lo}"
	
	# Validate that skill codes have definitions
	skill_codes = set(unit_payload.get('skill_codes', []))
	skill_definitions = unit_payload.get('skill_definitions', {})
	missing_definitions = skill_codes - set(skill_definitions.keys())
	if missing_definitions:
		return False, f"Missing skill definitions for codes: {missing_definitions}"
	
	return True, ""

def build_and_validate_unit_payload(
	course_name: str,
	unit_index: int,
	unit: dict,
	course_spec: dict,
	max_retries: int = 3
) -> dict:
	"""
	Build unit payload with retry logic and validation.
	Retries if payload is invalid, throws error after max retries.
	"""
	for attempt in range(1, max_retries + 1):
		try:
			unit_payload = build_unit_payload(course_name, unit_index, unit, course_spec)
			
			# Validate payload before returning
			is_valid, error_msg = validate_unit_payload(unit_payload)
			
			if is_valid:
				return unit_payload
			else:
				if attempt < max_retries:
					print(f"  ⚠️  Payload validation failed (attempt {attempt}/{max_retries}): {error_msg}")
					print(f"     Retrying payload construction...")
					time.sleep(0.5 * attempt)  # Small backoff between retries
				else:
					raise RuntimeError(
						f"Failed to build valid unit payload after {max_retries} attempts. "
						f"Last error: {error_msg}. "
						f"Payload summary: LOs={len(unit_payload.get('learning_objectives', []))}, "
						f"Skills={len(unit_payload.get('skill_codes', []))}, "
						f"Big Ideas={len(unit_payload.get('big_ideas', []))}"
					)
		except Exception as e:
			if attempt < max_retries:
				print(f"  ⚠️  Error building payload (attempt {attempt}/{max_retries}): {e}")
				print(f"     Retrying...")
				time.sleep(0.5 * attempt)
			else:
				raise RuntimeError(
					f"Failed to build unit payload after {max_retries} attempts. "
					f"Last error: {e}"
				)
	
	# Should never reach here, but just in case
	raise RuntimeError("Unexpected error in build_and_validate_unit_payload")

def build_unit_payload(course_name: str, unit_index: int, unit: dict, course_spec: dict) -> dict:
	"""
	Build unit payload respecting n-tier hierarchy:
	Course → Unit → Topic → Learning Objective → Skill Codes → Big Ideas
	
	Handles actual course JSON structure where:
	- skills are flat array with 'code' and 'description'
	- big_ideas have 'acronym' instead of 'id'
	- topics have 'skills' array (not 'suggested_subskill_codes')
	- learning_objectives have 'code' (not 'id') and 'skill_code'
	"""
	learning_objectives = []
	
	# Extract learning objectives from topics
	for topic in unit.get("topics", []):
		# Handle both structures: direct learning_objectives or nested
		topic_los = topic.get("learning_objectives", [])
		if not topic_los:
			continue
			
		for lo in topic_los:
			# Handle both 'id' and 'code' fields
			lo_id = lo.get("id") or lo.get("code") or lo.get("learning_objective_code", "")
			if not lo_id:
				continue
				
			learning_objectives.append({
				"id": lo_id,
				"description": lo.get("description", ""),
				"essential_knowledge": lo.get("essential_knowledge", []) or lo.get("essential_knowledge_codes", []) or []
			})

	# Extract skill codes from topics (handle 'skills' array or 'suggested_subskill_codes')
	skill_codes = set()
	
	# From unit-level skills
	for skill in unit.get("skills", []):
		code = skill.get("code") if isinstance(skill, dict) else skill
		if code:
			skill_codes.add(str(code))
	
	# From topic-level skills
	for topic in unit.get("topics", []):
		# Try 'skills' array first (actual structure)
		topic_skills = topic.get("skills", []) or topic.get("suggested_subskill_codes", [])
		for skill in topic_skills:
			code = skill.get("code") if isinstance(skill, dict) else skill
			if code:
				skill_codes.add(str(code))
		
		# Also extract from learning objectives' skill_code
		for lo in topic.get("learning_objectives", []):
			skill_code = lo.get("skill_code")
			if skill_code:
				skill_codes.add(str(skill_code))

	skill_codes = sorted(list(skill_codes))

	# Extract big ideas from unit and topics
	big_ideas = set()
	
	# From unit-level big_ideas
	for bi in unit.get("big_ideas", []):
		if isinstance(bi, str):
			big_ideas.add(str(bi))
		elif isinstance(bi, dict):
			bi_id = bi.get("acronym") or bi.get("id") or bi.get("name", "")
			if bi_id:
				big_ideas.add(str(bi_id))
	
	# From topic-level big_ideas
	for topic in unit.get("topics", []):
		# Handle direct big_ideas array
		topic_bis = topic.get("big_ideas", [])
		for bi in topic_bis:
			bi_id = bi if isinstance(bi, str) else (bi.get("acronym") or bi.get("id") or bi.get("name", ""))
			if bi_id:
				big_ideas.add(str(bi_id))
		
		# Handle enduring_understanding structure (can be dict or string)
		eu = topic.get("enduring_understanding")
		if eu:
			if isinstance(eu, dict):
				eu_acronym = eu.get("acronym", "")
				if eu_acronym:
					big_ideas.add(str(eu_acronym))
			elif isinstance(eu, str):
				# If it's a string, treat it as the acronym directly
				big_ideas.add(str(eu))

	big_ideas = sorted(list(big_ideas))

	# Build skill definitions from course_spec
	skill_definitions = {}
	# Handle flat skills array structure
	for skill in course_spec.get("skills", []):
		code = skill.get("code") or skill.get("subskill_name", "")
		if code in skill_codes:
			skill_definitions[str(code)] = {
				"category": skill.get("category") or skill.get("skill_name", ""),
				"description": skill.get("description") or skill.get("subskill_description", "")
			}
	
	# Handle nested skills structure (if present)
	for skill in course_spec.get("skills", []):
		if "subskills" in skill:
			for sub in skill.get("subskills", []):
				code = sub.get("subskill_name") or sub.get("code", "")
				if code in skill_codes:
					skill_definitions[str(code)] = {
						"category": skill.get("skill_name", ""),
						"description": sub.get("subskill_description", "")
					}

	# Build big idea definitions from course_spec
	big_idea_definitions = {}
	for bi in course_spec.get("big_ideas", []):
		bi_id = bi.get("acronym") or bi.get("id", "")
		if bi_id in big_ideas:
			big_idea_definitions[str(bi_id)] = {
				"name": bi.get("name") or bi.get("title", ""),
				"description": bi.get("description", "")
			}

	# Extract exam context
	mcq_exam_context = None
	for section in course_spec.get("exam_sections", []):
		if section.get("section") == "I":
			mcq_exam_context = {
				"question_type": section.get("question_type", ""),
				"exam_weighting": section.get("exam_weighting", ""),
				"timing": section.get("timing", ""),
				"descriptions": section.get("descriptions", []) or []
			}
			break

	return {
		"course": course_name,
		"unit": f"Unit {unit_index + 1}: {unit.get('title') or unit.get('name', '')}",
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

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1F\x7F]")

def _clean_model_text(s: str) -> str:
	# Strip common junk that breaks JSON parsing
	s = (s or "").strip()
	s = s.replace("\r", "")
	s = s.replace("\u0000", "")
	s = _CONTROL_CHARS_RE.sub("", s)
	return s

def generate_valid_mcqs(
	prompt_template,
	unit_payload: dict,
	total_questions: int,
	batch_size: int = 10,
	max_batches: int = 20,
	validate_items: bool = False,
	backoff_seconds: float = 0.75,
	debug_dir: str = "debug_mcq_outputs"
):
	"""
	Optimized MCQ generator:
	- Pre-serializes unit_payload JSON once per set
	- Pre-serializes schema JSON once per run (global)
	- Injects schema only into first batch per set
	- Validates at batch level by default (fast path)
	- Optionally validates each question (slower, for debugging)
	- Uses exponential backoff only on true model failures
	"""

	assert 0 < batch_size <= total_questions

	# Pre-serialize unit_payload once per set (big win)
	unit_payload_json = json.dumps(unit_payload, separators=(",", ":"))
	# Schema is pre-serialized globally as SCHEMA_JSON

	ensure_dir(debug_dir)

	all_questions = []
	batches_attempted = 0
	consecutive_failures = 0
	schema_injected = False  # Track if schema has been injected for this set

	# Useful for avoiding infinite loops if the model keeps returning invalid JSON
	min_batches_needed = (total_questions + batch_size - 1) // batch_size
	# Give yourself some slack beyond the theoretical minimum
	effective_max_batches = max(max_batches, min_batches_needed + 5)

	print(f"    Starting MCQ generation: total={total_questions}, batch_size={batch_size}")

	while len(all_questions) < total_questions and batches_attempted < effective_max_batches:
		batches_attempted += 1
		remaining = total_questions - len(all_questions)
		n = min(batch_size, remaining)

		print(f"      Batch {batches_attempted}: requesting n={n} (have {len(all_questions)}/{total_questions})")

		# Inject schema only into first batch per set
		# Subsequent batches rely on structure lock-in from first batch
		if schema_injected:
			# For subsequent batches, use schema reference only (not full schema)
			prompt = prompt_template.render(
				num_questions=n,
				topic_payload=unit_payload_json,
				schema="[Schema structure locked from first batch - maintain same format]"
			)
		else:
			# First batch gets full schema
			prompt = prompt_template.render(
				num_questions=n,
				topic_payload=unit_payload_json,
				schema=SCHEMA_JSON
			)
			schema_injected = True

		try:
			response = client.models.generate_content(
				model=MODEL,
				contents=prompt,
				config={"response_mime_type": "application/json"}
			)
		except Exception as e:
			# Only backoff on true model failures (network, API errors)
			consecutive_failures += 1
			backoff_time = backoff_seconds * (2 ** min(consecutive_failures - 1, 4))
			print(f"      Model call failed ({type(e).__name__}). failures={consecutive_failures}, backing off {backoff_time:.2f}s")
			time.sleep(backoff_time)
			continue

		raw = _clean_model_text(getattr(response, "text", ""))

		# Parse JSON
		try:
			parsed = json.loads(raw)
		except json.JSONDecodeError as e:
			consecutive_failures += 1
			print(f"      Invalid JSON (parse error). failures={consecutive_failures}")
			
			# Save for inspection
			debug_path = os.path.join(debug_dir, f"batch_{batches_attempted}_need_{n}_parse_error.txt")
			with open(debug_path, "w", encoding="utf-8") as f:
				f.write(f"JSON Parse Error: {e}\n\nRaw response:\n{raw}")
			
			# No backoff for JSON parse errors - likely prompt issue
			continue

		# Batch-level schema validation (fast path)
		# Only validate if we have a valid structure
		if not isinstance(parsed, dict):
			consecutive_failures += 1
			print(f"      Invalid response structure (not a dict). failures={consecutive_failures}")
			debug_path = os.path.join(debug_dir, f"batch_{batches_attempted}_need_{n}_structure_error.txt")
			with open(debug_path, "w", encoding="utf-8") as f:
				f.write(f"Response is not a dict. Type: {type(parsed)}\n\nContent:\n{json.dumps(parsed, indent=2)}")
			continue
		
		try:
			validate(instance=parsed, schema=MCQ_SCHEMA)
		except ValidationError as e:
			consecutive_failures += 1
			print(f"      Schema validation failed. failures={consecutive_failures}")
			# Show the path where validation failed
			error_path = " -> ".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"
			print(f"      Error at: {error_path}")
			print(f"      Message: {e.message[:200]}")

			# Save for inspection
			debug_path = os.path.join(debug_dir, f"batch_{batches_attempted}_need_{n}_schema_error.txt")
			with open(debug_path, "w", encoding="utf-8") as f:
				f.write(f"Schema Validation Error: {e}\n\nParsed JSON:\n{json.dumps(parsed, indent=2)}")

			# No backoff for schema errors - likely model output issue
			continue

		# Reset failure counter on success
		consecutive_failures = 0

		qs = parsed.get("questions", [])
		if not isinstance(qs, list) or len(qs) == 0:
			print("      Parsed but contained no questions; retrying batch")
			continue

		added = 0
		for q in qs:
			if validate_items:
				try:
					validate(instance=q, schema=QUESTION_SCHEMA)
				except Exception:
					continue

			all_questions.append(q)
			added += 1

			if len(all_questions) >= total_questions:
				break

		print(f"      Added {added}. total now {len(all_questions)}/{total_questions}")

	if len(all_questions) < total_questions:
		raise RuntimeError(
			f"Failed to generate {total_questions} MCQs "
			f"(got {len(all_questions)} after {batches_attempted} batches)"
		)

	print(f"    Finished MCQ generation: {len(all_questions)} questions")
	return {"questions": all_questions[:total_questions]}


# -----------------------
# LOAD STATIC ASSETS
# -----------------------

MCQ_SCHEMA = load_json(MCQ_SCHEMA_PATH)
QUESTION_SCHEMA = MCQ_SCHEMA["properties"]["questions"]["items"]
MCQ_HTML_TEMPLATE = Template(load_text(MCQ_HTML_TEMPLATE_PATH))
MCQ_PROMPT_TEMPLATE = Template(load_text(MCQ_PROMPT_PATH))

# Pre-serialize schema once per run (performance optimization)
SCHEMA_JSON = json.dumps(MCQ_SCHEMA, separators=(",", ":"))

# -----------------------
# MAIN LOOP (ROUND-ROBIN BY SET)
# -----------------------

for course in AP_COURSES:
	print(f"\n📘 Processing course: {course}")

	course_spec = load_json(f"{BASE_SPEC_DIR}/content/{course}.json")
	course_name = course_spec.get("name", course)
	units = course_spec.get("units", [])

	print(f"📦 Total units: {len(units)}")
	print(f"📦 Sets per unit: {NUM_SETS_PER_UNIT}")

	# OUTER LOOP: SET INDEX
	for set_index in range(NUM_SETS_PER_UNIT):
		print("\n======================================")
		print(f"🧩 STARTING SET {set_index + 1}/{NUM_SETS_PER_UNIT}")
		print("======================================")

		# INNER LOOP: UNITS
		for unit_index, unit in enumerate(units):
			unit_title = unit.get("title") or unit.get("name") or f"Unit {unit_index + 1}"

			print("\n--------------------------------------")
			print(f"📄 UNIT {unit_index + 1}/{len(units)}")
			print(f"📘 {unit_title}")
			print(f"🧩 SET {set_index + 1}")
			print("--------------------------------------")

			# Check if output already exists - only skip in this case
			output_dir = f"{OUTPUT_BASE_DIR}/{course}/unit_{unit_index + 1}/mcqs"
			ensure_dir(output_dir)
			
			if mcq_set_exists(course, unit_index, set_index):
				print(f"  ⏭️  Skipping — already exists (unit {unit_index + 1}, set {set_index + 1})")
				continue

			# Build and validate payload with retry logic
			# This will throw an error if payload can't be validated after retries
			print(f"  🔍 Building and validating unit payload...")
			unit_payload = build_and_validate_unit_payload(
				course_name,
				unit_index,
				unit,
				course_spec,
				max_retries=3
			)

			print(f"  ✅ Payload validated successfully")
			print(f"  🧠 Payload summary:")
			print(f"     • Learning objectives: {len(unit_payload['learning_objectives'])}")
			print(f"     • Skill codes: {len(unit_payload['skill_codes'])}")
			print(f"     • Big ideas: {len(unit_payload['big_ideas'])}")
			print(f"     • Skill definitions: {len(unit_payload.get('skill_definitions', {}))}")
			print(f"     • Big idea definitions: {len(unit_payload.get('big_idea_definitions', {}))}")

			print(f"  🚀 Generating MCQs for this unit/set")

			mcq_json = generate_valid_mcqs(
				MCQ_PROMPT_TEMPLATE,
				unit_payload,
				QUESTIONS_PER_SET,
				batch_size=10,  # Safe batch size (can increase to 15-20 if needed)
				validate_items=False  # Batch-level validation is sufficient
			)

			print("  ✂️ Cleaning answer choices")
			for q in mcq_json["questions"]:
				q["choices"] = [strip_choice_labels(c) for c in q["choices"]]

			print("  🧱 Converting table stimuli (if any)")
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

			print(f"  💾 Saved → {output_path}")

print("\n🎉 All MCQ sets generated successfully (round-robin mode).")

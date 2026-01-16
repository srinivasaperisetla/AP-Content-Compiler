'''
Still need to implement:
- Handling the variance of AP courses for example AP Lang and AP Lit are different
- Handling of Empty Values fore example the LLM might trip and produce and entry with empty values we have to remove them
'''

import os
import json
import re
import pdfplumber
from dotenv import load_dotenv
from google import genai
from google.genai import types
import certifi
import pathlib
from pathlib import Path

os.environ["SSL_CERT_FILE"] = certifi.where()

MODEL = "gemini-2.5-flash"

#initialize gemini client
print("Initializing environment...")
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
assert API_KEY, "Missing GEMINI_API_KEY"
client = genai.Client(api_key=API_KEY)
print("Gemini client initialized")

# Helper Functions
def normalize_section(section):
	section = section.strip()
	if not section:
		return None

	section = section.replace("Section", "").strip()

	# Roman numerals only
	match = re.match(r"^(I|II|III|IV|V|VI|IB)$", section)
	if match:
		return match.group(1)

	return None

def normalize_question_type(q_type):
	q = q_type.lower()
	if "multiple" in q:
		return "Multiple Choice"
	if "free" in q:
		return "Free Response"
	if "short" in q:
		return "Short Answer"
	if "document" in q:
		return "Document-Based Question"
	if "project" in q:
		return "Individual Student Project"
	return q_type.strip()



# functions for each of the sections
def get_course_skills(pdf_path):
	prompt_path = Path("utils/prompts/skills_prompt.txt")
	skills_prompt = prompt_path.read_text(encoding="utf-8").strip()

	filepath = Path(pdf_path)

	response = client.models.generate_content(
		model="gemini-2.5-pro",
		contents=[
			types.Part.from_bytes(
				data=filepath.read_bytes(),
				mime_type="application/pdf",
			),
			skills_prompt,
		],
	)

	raw_text = response.text
	print("\n[skills] Gemini normalized output:\n")
	print(raw_text)

	skills_map = {}

	for line in raw_text.splitlines():
		line = line.strip()
		if not line:
			continue

		cols = line.split("\t")
		if len(cols) != 5:
			print(f"[skills][WARN] Expected 5 columns, got {len(cols)}:")
			print(line)
			continue

		category_label = cols[0].strip()  # kept if you want later
		skill_name = cols[1].strip()
		skill_desc = cols[2].strip()
		sub_code = cols[3].strip()
		sub_desc = cols[4].strip()

		if skill_name not in skills_map:
			skills_map[skill_name] = {
				"skill_name": skill_name,
				"skill_description": skill_desc,
				"subskills": [],
			}

		# Fill description once if present
		if skill_desc and not skills_map[skill_name]["skill_description"]:
			skills_map[skill_name]["skill_description"] = skill_desc

		skills_map[skill_name]["subskills"].append({
			"subskill_name": sub_code,
			"subskill_description": sub_desc,
		})

	skills = list(skills_map.values())
	print(f"\n[skills] Parsed {len(skills)} skill categories")
	return skills


def get_big_ideas(pdf_path):
	prompt_path = Path("utils/prompts/big_ideas_prompt.txt")
	big_ideas_prompt = prompt_path.read_text(encoding="utf-8").strip()

	filepath = Path(pdf_path)
	if not filepath.exists():
		print(f"[big_ideas] File not found: {pdf_path}")
		return []

	response = client.models.generate_content(
		model="gemini-2.5-pro",
		contents=[
			types.Part.from_bytes(
				data=filepath.read_bytes(),
				mime_type="application/pdf",
			),
			big_ideas_prompt,
		],
	)

	raw_text = response.text or ""
	print("\n[big_ideas] Gemini normalized output:\n")
	print(raw_text)

	big_ideas = []
	seen = set()

	for line in raw_text.splitlines():
		line = line.strip()
		if not line:
			continue

		cols = line.split("\t")
		if len(cols) != 3:
			print(f"[big_ideas][WARN] Expected 3 columns, got {len(cols)}:")

		if len(cols) == 2:
			big_id = cols[0]
			name = cols[0]
			desc = cols[1]

		# Handle 3-column case (e.g. AP Stats)
		elif len(cols) == 3:
			big_id = cols[0]
			name = cols[1]
			desc = cols[2]

		else:
			print(f"[big_ideas][WARN] Expected 2 or 3 columns, got {len(cols)}:")
			print(line)
			continue

		key = (big_id, name, desc)
		if key in seen:
			continue
		seen.add(key)

		big_ideas.append({
			"id": big_id,
			"name": name,
			"description": desc,
		})

	print(f"\n[big_ideas] Parsed {len(big_ideas)} big ideas")
	return big_ideas

def get_exam_sections(pdf_path):
	prompt_path = Path("utils/prompts/exam_sections_prompt.txt")
	exam_sections_prompt = prompt_path.read_text(encoding="utf-8").strip()

	filepath = Path(pdf_path)
	if not filepath.exists():
		print(f"[exam_sections] File not found: {pdf_path}")
		return []

	response = client.models.generate_content(
		model="gemini-2.5-pro",
		contents=[
			types.Part.from_bytes(
				data=filepath.read_bytes(),
				mime_type="application/pdf",
			),
			exam_sections_prompt,
		],
	)

	raw_text = response.text or ""
	print("\n[exam_sections] Gemini normalized output:\n")
	print(raw_text)

	sections_map = {}

	for line in raw_text.splitlines():
		line = line.strip()
		if not line:
			continue

		cols = [c.strip() for c in re.split(r"\t|\s{2,}", line)]

		if len(cols) < 2:
			continue

		while len(cols) < 6:
			cols.append("")

		raw_section, q_type, num_qs, weight, timing, desc = cols[:6]

		if raw_section.lower().startswith("section") and q_type.lower().startswith("question"):
			continue

		section = normalize_section(raw_section)
		if not section:
			continue

		q_type = normalize_question_type(q_type)

		if section not in sections_map:
			sections_map[section] = {
				"section": section,
				"question_type": q_type,
				"Number of Questions": num_qs,
				"exam_weighting": "",
				"timing": "",
				"descriptions": [],
			}

		entry = sections_map[section]

		if not entry["question_type"] and q_type:
			entry["question_type"] = q_type

		if not entry["exam_weighting"] and weight:
			entry["exam_weighting"] = weight

		if not entry["timing"] and timing:
			entry["timing"] = timing

		if desc:
			entry["descriptions"].append(desc)

	exam_sections = list(sections_map.values())

	print(f"\n[exam_sections] Parsed {len(exam_sections)} exam sections")
	return exam_sections

def get_task_verbs(pdf_path):
	prompt_path = Path("utils/prompts/task_verbs_prompt.txt")
	task_verbs_prompt = prompt_path.read_text(encoding="utf-8").strip()

	filepath = Path(pdf_path)
	if not filepath.exists():
		print(f"[exam_sections] File not found: {pdf_path}")
		return []

	response = client.models.generate_content(
		model="gemini-2.5-pro",
		contents=[
			types.Part.from_bytes(
				data=filepath.read_bytes(),
				mime_type="application/pdf",
			),
			task_verbs_prompt,
		],
	)

	raw_text = response.text or ""
	print("\n[task verbs] Gemini normalized output:\n")
	print(raw_text)

	task_verbs = []
	seen = set()

	for line in raw_text.splitlines():
		line = line.strip()
		if not line:
			continue

		# Prefer TAB, fallback to 2+ spaces
		if "\t" in line:
			cols = [c.strip() for c in line.split("\t")]
		else:
			cols = [c.strip() for c in re.split(r"\s{2,}", line)]

		if len(cols) < 2:
			print(f"[task_verbs][WARN] Expected 2 columns, got {len(cols)}:")
			print(line)
			continue

		verb = cols[0].strip()
		desc = cols[1].strip()

		# Skip header-ish rows
		if verb.lower() in {"verb", "task verb"} and "description" in desc.lower():
			continue

		key = (verb, desc)
		if key in seen:
			continue
		seen.add(key)

		task_verbs.append({
			"verb": verb,
			"description": desc,
		})

	print(f"\n[task_verbs] Parsed {len(task_verbs)} task verbs")
	return task_verbs

def get_units(pdf_path):
	prompt_path = Path("utils/prompts/units_prompt.txt")
	units_prompt = prompt_path.read_text(encoding="utf-8").strip()

	filepath = Path(pdf_path)
	if not filepath.exists():
		print(f"[units] File not found: {pdf_path}")
		return None

	response = client.models.generate_content(
		model="gemini-2.5-pro",
		contents=[
			types.Part.from_bytes(
				data=filepath.read_bytes(),
				mime_type="application/pdf",
			),
			units_prompt,
		],
	)

	raw_text = (response.text or "").strip()
	print("\n[units] Gemini normalized output:\n")
	print(raw_text)

	def _split_cols(line):
		# Prefer real TABs; fallback to 2+ spaces
		if "\t" in line:
			return [c.strip() for c in line.split("\t")]
		return [c.strip() for c in re.split(r"\s{2,}", line)]

	def _safe_get(cols, idx):
		if idx < 0 or idx >= len(cols):
			return ""
		return cols[idx].strip()

	def _infer_unit_id_from_filename(path_str):
		# e.g., unit_1_ap_statistics.pdf -> "1"
		m = re.search(r"unit_(\d+)_", Path(path_str).name, flags=re.IGNORECASE)
		return m.group(1) if m else ""

	unit = {
		"id": "",
		"name": "",
		"developing_understanding": "",
		"building_practices": "",
		"preparing_for_exam": "",
		"topics": [],
	}

	topics_map = {}
	current_topic_id = None

	def _get_or_create_topic(topic_id, topic_name):
		nonlocal current_topic_id
		tid = (topic_id or "").strip()
		tname = (topic_name or "").strip()

		# If topic_id missing, make a stable synthetic key so we don't lose content.
		# This is deterministic and local to this unit.
		if not tid:
			tid = f"_topic_{len(topics_map) + 1}"

		if tid not in topics_map:
			topics_map[tid] = {
				"id": "" if tid.startswith("_topic_") else tid,
				"name": tname,
				"suggested_subskill_codes": [],
				"big_ideas": [],
				"learning_objectives": [],
			}
			unit["topics"].append(topics_map[tid])

		# If name is missing earlier but appears later, fill it once.
		if tname and not topics_map[tid]["name"]:
			topics_map[tid]["name"] = tname

		current_topic_id = tid
		return topics_map[tid]

	def _get_or_create_lo(topic_obj, lo_id, lo_desc):
		loid = (lo_id or "").strip()
		lodesc = (lo_desc or "").strip()

		if "_lo_index" not in topic_obj:
			topic_obj["_lo_index"] = {}

		# Key ONLY by lo_id (not description)
		key = loid

		if key not in topic_obj["_lo_index"]:
			lo_obj = {
				"id": loid,
				"description": lodesc,
				"essential_knowledge": [],
			}
			topic_obj["learning_objectives"].append(lo_obj)
			topic_obj["_lo_index"][key] = lo_obj
		else:
			lo_obj = topic_obj["_lo_index"][key]
			# If description was missing earlier, fill it now
			if lodesc and not lo_obj["description"]:
				lo_obj["description"] = lodesc

		return lo_obj


	def _cleanup_internal_indexes():
		for t in unit["topics"]:
			if "_lo_index" in t:
				del t["_lo_index"]

	seen_subskills = set()
	seen_bigideas = set()
	seen_los = set()
	seen_eks = set()

	for line in raw_text.splitlines():
		line = line.strip()
		if not line:
			continue

		cols = _split_cols(line)
		record_type = _safe_get(cols, 0).upper()

		if record_type == "UNIT":
			# UNIT<TAB>unit_id<TAB>unit_name<TAB>developing_understanding<TAB>building_practices<TAB>preparing_for_exam
			unit_id = _safe_get(cols, 1)
			unit_name = _safe_get(cols, 2)
			dev = _safe_get(cols, 3)
			build = _safe_get(cols, 4)
			prep = _safe_get(cols, 5)

			# Fill once; allow later lines to fill blanks if earlier missing
			if unit_id and not unit["id"]:
				unit["id"] = unit_id
			if unit_name and not unit["name"]:
				unit["name"] = unit_name
			if dev and not unit["developing_understanding"]:
				unit["developing_understanding"] = dev
			if build and not unit["building_practices"]:
				unit["building_practices"] = build
			if prep and not unit["preparing_for_exam"]:
				unit["preparing_for_exam"] = prep

		elif record_type == "TOPIC":
			# TOPIC<TAB>topic_id<TAB>topic_name
			topic_id = _safe_get(cols, 1)
			topic_name = _safe_get(cols, 2)
			_get_or_create_topic(topic_id, topic_name)

		elif record_type == "SUBSKILL":
			# SUBSKILL<TAB>topic_id<TAB>subskill_code
			topic_id = _safe_get(cols, 1)
			subskill_code = _safe_get(cols, 2)

			topic_obj = _get_or_create_topic(topic_id, "")
			if subskill_code:
				key = (topic_obj.get("id", ""), subskill_code)
				if key not in seen_subskills:
					seen_subskills.add(key)
					topic_obj["suggested_subskill_codes"].append(subskill_code)

		elif record_type == "BIGIDEA":
			topic_id = _safe_get(cols, 1)
			big_id = _safe_get(cols, 2)
			big_desc = _safe_get(cols, 3)

			# ❌ Ignore unit-level big ideas (no topic_id)
			if not topic_id:
				continue

			topic_obj = _get_or_create_topic(topic_id, "")
			key = (topic_obj.get("id", ""), big_id, big_desc)
			if key not in seen_bigideas:
				seen_bigideas.add(key)
				topic_obj["big_ideas"].append({
					"id": big_id,
					"description": big_desc,
				})

		elif record_type == "LO":
			# LO<TAB>topic_id<TAB>learning_objective_id<TAB>learning_objective_description
			topic_id = _safe_get(cols, 1)
			lo_id = _safe_get(cols, 2)
			lo_desc = _safe_get(cols, 3)

			topic_obj = _get_or_create_topic(topic_id, "")
			key = (topic_obj.get("id", ""), lo_id)
			if key not in seen_los:
				seen_los.add(key)
				_get_or_create_lo(topic_obj, lo_id, lo_desc)

		elif record_type == "EK":
			# EK<TAB>topic_id<TAB>learning_objective_id<TAB>essential_knowledge_id<TAB>essential_knowledge_description
			topic_id = _safe_get(cols, 1)
			lo_id = _safe_get(cols, 2)  # may be empty
			ek_id = _safe_get(cols, 3)
			ek_desc = _safe_get(cols, 4)

			topic_obj = _get_or_create_topic(topic_id, "")

			# If there is no LO in the doc, keep LO id blank and attach EK under a blank LO bucket
			lo_obj = _get_or_create_lo(topic_obj, lo_id, "")

			key = (topic_obj.get("id", ""), lo_obj.get("id", ""), ek_id, ek_desc)
			if key not in seen_eks:
				seen_eks.add(key)
				lo_obj["essential_knowledge"].append({
					"id": ek_id,
					"description": ek_desc,
				})

		else:
			# Ignore anything that doesn't match the typed TSV format
			continue

	_cleanup_internal_indexes()

	unit["topics"] = [
		t for t in unit["topics"]
		if t["id"] and (t["learning_objectives"] or t["big_ideas"] or t["suggested_subskill_codes"])
	]

	# Final fallback for unit id if Gemini didn't provide it
	if not unit["id"]:
		unit["id"] = _infer_unit_id_from_filename(pdf_path)

	print(f"\n[units] Parsed unit id={unit['id']} topics={len(unit['topics'])}")

	return unit



TEMPLATE_PATH = "ap_specs/1template.json"
OUTPUT_DIR = "utils/content"
AP_COURSES = [
	# "ap_statistics",
	# "ap_physics_1"
	# "ap_biology",
	# "ap_english_language_and_composition",
	# "ap_african_american_studies",
	# "ap_calculus_ab_and_bc",
	"ap_environmental_science",
	"ap_chemistry",
	# "ap_macroeconomics",
	# "ap_human_geography",
]

def main():
	for course in AP_COURSES:
		base = f"ap_specs/{course}"
		output_path = f"{OUTPUT_DIR}/{course}.json"

		with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
			result = json.load(f)

		result["name"] = course.replace("_", " ").title()

		skills_pdf = f"{base}/skills_{course}.pdf"
		big_ideas_pdf = f"{base}/big_ideas_{course}.pdf"
		exam_sections_pdf = f"{base}/exam_sections_{course}.pdf"
		task_verbs_pdf = f"{base}/task_verbs_{course}.pdf"

		result["skills"] = get_course_skills(skills_pdf)
		result["big_ideas"] = get_big_ideas(big_ideas_pdf)
		result["exam_sections"] = get_exam_sections(exam_sections_pdf)
		result["task_verbs"] = get_task_verbs(task_verbs_pdf)
	
		result["units"] = []
		units_dir = f"{base}/units"

		if os.path.isdir(units_dir):
			for filename in sorted(os.listdir(units_dir)):
				if not filename.lower().endswith(".pdf"):
					continue  # skip .DS_Store and any junk files

				unit_pdf_path = f"{units_dir}/{filename}"
				print(f"[units] Processing {unit_pdf_path}")

				unit_data = get_units(unit_pdf_path)
				if unit_data:
					result["units"].append(unit_data)
		else:
			print(f"[units][WARN] No units directory found for {course}")

		os.makedirs(OUTPUT_DIR, exist_ok=True)
		with open(output_path, "w", encoding="utf-8") as f:
			json.dump(result, f, indent=2, ensure_ascii=False)

		print(f"Wrote {output_path}")

if __name__ == "__main__":
	main()




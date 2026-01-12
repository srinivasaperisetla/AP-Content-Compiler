'''
Problems right now:
- Retry logic for only the questions that are invalid and that is missed so that the LLM can fix them easily instead of regenerating the entire set. 
- Trim Unit Context as well as the prompt to save on tokens.


- Optional Split generation into: planning pass (topics → question plan) then generation pass (plan → TSV)
- Optional Parallelize of units
'''


import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from jinja2 import Template
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()

MODEL = "gemini-2.5-pro"

NUM_SETS_PER_UNIT = 20
QUESTIONS_PER_SET = 25

MAX_RETRIES_PER_SET = 4

AP_COURSES = [
	"ap_statistics",
]

CONTENT_DIR = Path("utils/content")
PROMPT_PATH = Path("utils/prompts/mcq_prompt.txt")
HTML_TEMPLATE_PATH = Path("utils/templates/mcq.html")
OUTPUT_DIR = Path("output")

DEBUG = True
MAX_PROMPT_CHARS = 6000


# -----------------------
# Debug logging
# -----------------------
def _ts() -> str:
	return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
	print(f"[{_ts()}] {msg}")


def log_block(title: str, text: str, max_chars: int) -> None:
	if not DEBUG:
		return
	log(f"{title} (chars={len(text)} showing up to {max_chars})")
	print(text[:max_chars])
	if len(text) > max_chars:
		print("... [TRUNCATED] ...")


def log_context(course: str, unit_index: int, unit_name: str, set_index: int) -> str:
	return f"{course} | Unit {unit_index + 1}: {unit_name} | Set {set_index + 1}/{NUM_SETS_PER_UNIT}"


# -----------------------
# Init Gemini client
# -----------------------
def init_client() -> genai.Client:
	load_dotenv()
	api_key = os.getenv("GEMINI_API_KEY")
	assert api_key, "Missing GEMINI_API_KEY"
	google_client = genai.Client(api_key=api_key)
	print("Gemini client initialized")
	return google_client

# -----------------------
# Utilities
# -----------------------
def load_json(path: Path) -> dict:
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def load_text(path: Path) -> str:
	with path.open("r", encoding="utf-8") as f:
		return f.read()


def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def safe_join_lines(lines: List[str]) -> str:
	return "\n".join([ln.rstrip() for ln in lines if ln and ln.strip()])


def normalize_whitespace(s: str) -> str:
	return re.sub(r"[ \t]+", " ", (s or "").strip())


def pipe_table_to_html(table_text: str) -> str:
	if not table_text:
		return ""

	# Normalize escaped newlines
	text = table_text.replace("\\n", "\n")
	lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

	# Strict structural validation
	if not is_strict_pipe_table(lines):
		return ""

	# Parse header and data
	headers = [h.strip() for h in lines[0].strip("|").split("|")]
	data_lines = lines[2:]  # skip header + separator

	if not headers or not data_lines:
		return ""

	# Build HTML
	html = "<table class='data-table'><thead><tr>"
	for h in headers:
		html += f"<th>{h}</th>"
	html += "</tr></thead><tbody>"

	for ln in data_lines:
		cells = [c.strip() for c in ln.strip("|").split("|")]
		if len(cells) != len(headers):
			return ""  # hard fail, triggers repair
		html += "<tr>"
		for cell in cells:
			html += f"<td>{cell}</td>"
		html += "</tr>"

	html += "</tbody></table>"
	return html




# -----------------------
# Validation
# -----------------------
def is_valid_svg(svg: str) -> bool:
	if not svg:
		return False

	s = svg.strip()

	if not (s.startswith("<svg") and s.endswith("</svg>")):
		return False

	if "<script" in s.lower():
		return False

	if s.count("<text") > 8:
		return False

	if 'font-size="' in s:
		sizes = re.findall(r'font-size="(\d+)"', s)
		if any(int(sz) < 12 for sz in sizes):
			return False

	return True


def is_strict_pipe_table(lines: List[str]) -> bool:
	if len(lines) < 3:
		return False

	# all rows must start/end with |
	if not all(ln.startswith("|") and ln.endswith("|") for ln in lines):
		return False

	# separator row must be dashes
	sep = lines[1].strip("|").split("|")
	if not all(set(cell.strip()) <= {"-"} for cell in sep):
		return False

	# consistent column count
	col_count = len(lines[0].split("|"))
	return all(len(ln.split("|")) == col_count for ln in lines)


def validate_rows_individually(
	rows: List[List[str]],
	constraints: Dict[str, Any],
	context_label: str
) -> Tuple[List[dict], List[dict]]:
	"""
	Returns:
	  valid_questions: List[question_dict]
	  invalid_reports: List[{row_index, reason, detail}]
	"""
	valid_questions = []
	invalid_reports = []

	for row_i, cols in enumerate(rows, start=1):
		error = validate_tsv_row(cols)
		if error:
			invalid_reports.append({
				"row_index": row_i,
				"reason": "row_invalid",
				"detail": error
			})
			continue

		qid, diff, skills, los, idx, qtext, A, B, C, D, stim_type, stim_payload = [
			c.strip() for c in cols
		]

		skill_codes = [s for s in skills.split(",") if s]
		if any(s not in constraints["allowed_skill_codes"] for s in skill_codes):
			invalid_reports.append({
				"row_index": row_i,
				"reason": "skill_not_allowed",
				"detail": skills
			})
			continue

		lo_ids = [s for s in los.split(",") if s]
		if any(lo not in constraints["allowed_lo_ids"] for lo in lo_ids):
			invalid_reports.append({
				"row_index": row_i,
				"reason": "lo_not_allowed",
				"detail": los
			})
			continue

		if stim_type == "svg" and not is_valid_svg(stim_payload):
			invalid_reports.append({
				"row_index": row_i,
				"reason": "svg_invalid",
				"detail": ""
			})
			continue

		if stim_type == "table":
			if "<table" not in pipe_table_to_html(stim_payload):
				invalid_reports.append({
					"row_index": row_i,
					"reason": "table_invalid",
					"detail": ""
				})
				continue

		stimulus = None
		if stim_type == "table":
			html_table = pipe_table_to_html(stim_payload)
			if "<table" not in html_table:
				continue
			stimulus = {"type": "table", "content": html_table}

		elif stim_type == "svg":
			if not is_valid_svg(stim_payload):
				continue
			stimulus = {"type": "svg", "content": stim_payload}

		valid_questions.append({
			"id": qid,
			"difficulty": diff,
			"skill_codes": skill_codes,
			"aligned_lo_ids": lo_ids,
			"correct_choice_index": int(idx),
			"question": qtext,
			"choices": [A, B, C, D],
			"stimulus": stimulus
		})


	return valid_questions, invalid_reports

def summarize_errors(invalid_reports: List[dict]) -> str:
	counts = {}
	for r in invalid_reports:
		counts[r["reason"]] = counts.get(r["reason"], 0) + 1

	lines = []
	for k, v in sorted(counts.items()):
		lines.append(f"- {k}: {v}")

	return "\n".join(lines)


def validate_tsv_row(cols: List[str]) -> Optional[str]:
	if len(cols) != 12:
		return "Wrong column count"

	qid, diff, skills, los, idx, qtext, A, B, C, D, stim_type, stim_payload = cols

	if not qid.strip():
		return "Empty question_id"

	if diff not in {"easy", "medium", "hard"}:
		return "Invalid difficulty"

	if not skills.strip():
		return "Empty skill_codes"

	if not los.strip():
		return "Empty learning_objective_ids"

	if not qtext.strip():
		return "Empty question text"

	if not all([A.strip(), B.strip(), C.strip(), D.strip()]):
		return "Empty choice"

	if stim_type not in {"none", "svg", "table"}:
		return "Invalid stimulus_type"

	try:
		i = int(idx)
		if i not in {0, 1, 2, 3}:
			return "Invalid correct_index"
	except:
		return "Non-integer correct_index"

	if stim_type == "none" and stim_payload.strip():
		return "stimulus_payload must be empty when stimulus_type=none"

	if stim_type != "none" and not stim_payload.strip():
		return "Missing stimulus_payload"

	return None



# -----------------------
# Lookups
# -----------------------
def build_skill_lookup(course_spec: dict) -> Dict[str, Dict[str, str]]:
	out = {}
	for skill_cat in course_spec.get("skills", []):
		cat_name = skill_cat.get("skill_name", "")
		for sub in skill_cat.get("subskills", []):
			code = str(sub.get("subskill_name", "")).strip()
			if not code:
				continue
			out[code] = {
				"category": cat_name,
				"description": sub.get("subskill_description", "")
			}
	return out


def build_big_idea_lookup(course_spec: dict) -> Dict[str, Dict[str, str]]:
	out = {}
	for bi in course_spec.get("big_ideas", []):
		bi_id = str(bi.get("id", "")).strip()
		if not bi_id:
			continue
		out[bi_id] = {
			"name": bi.get("name", ""),
			"description": bi.get("description", "")
		}
	return out


# -----------------------
# Unit context builder
# -----------------------
def build_unit_context(
	course_spec: dict,
	unit: dict,
	unit_index: int,
	skill_lookup: Dict[str, Dict[str, str]],
	big_idea_lookup: Dict[str, Dict[str, str]],
	max_ek_per_lo: int = 0,              # keep OFF for compression
	include_skill_descriptions: bool = True,
	include_course_big_ideas: bool = True,
	include_topic_big_idea_descriptions: bool = True,
	max_topic_name_chars: int = 90,
	max_lo_desc_chars: int = 200,
	max_skill_desc_chars: int = 140,
) -> Tuple[str, Dict[str, Any]]:

	def trunc(s: str, n: int) -> str:
		s = normalize_whitespace(s or "")
		return s if len(s) <= n else s[: n - 3] + "..."

	allowed_skill_codes: set = set()
	allowed_lo_ids: set = set()

	used_skill_codes: set = set()

	# Topic-level big ideas (e.g., VAR-1, UNC-1, VAR-2)
	used_topic_big_ids: set = set()
	topic_big_desc: Dict[str, str] = {}

	# Course-level big idea categories (e.g., VAR, UNC, DAT)
	used_course_big_ids: set = set()
	course_big_desc: Dict[str, str] = {}
	course_big_name: Dict[str, str] = {}

	topic_blocks: List[str] = []

	# -----------------------
	# Extract per-topic content
	# -----------------------
	for topic in unit.get("topics", []) or []:
		topic_id = str(topic.get("id", "")).strip()
		topic_name = trunc(topic.get("name", ""), max_topic_name_chars)

		# Skills for topic
		ssc = [
			str(x).strip()
			for x in (topic.get("suggested_subskill_codes") or [])
			if str(x).strip()
		]
		for code in ssc:
			allowed_skill_codes.add(code)
			used_skill_codes.add(code)

		# Topic-level big ideas for topic
		bis: List[str] = []
		for bi in (topic.get("big_ideas") or []):
			bi_id = str(bi.get("id", "")).strip()
			bi_d = normalize_whitespace(bi.get("description", "") or "")
			if not bi_id:
				continue

			bis.append(bi_id)
			used_topic_big_ids.add(bi_id)

			# keep the (usually short) description from topic if present
			if include_topic_big_idea_descriptions and bi_d and bi_id not in topic_big_desc:
				topic_big_desc[bi_id] = bi_d

			# also track course-level category prefix: VAR / UNC / DAT
			if "-" in bi_id:
				prefix = bi_id.split("-", 1)[0]
				if prefix:
					used_course_big_ids.add(prefix)

		# Learning objectives (core)
		lo_lines: List[str] = []
		for lo in (topic.get("learning_objectives") or []):
			lo_id = str(lo.get("id", "")).strip()
			if not lo_id:
				continue

			allowed_lo_ids.add(lo_id)
			lo_desc = trunc(lo.get("description", ""), max_lo_desc_chars)
			lo_lines.append(f"{lo_id}: {lo_desc}")

			# Optional EKs (off by default)
			if max_ek_per_lo and (lo.get("essential_knowledge") or []):
				for ek in (lo.get("essential_knowledge") or [])[:max_ek_per_lo]:
					ek_id = str(ek.get("id", "")).strip()
					ek_desc = trunc(ek.get("description", ""), max_lo_desc_chars)
					if ek_id and ek_desc:
						lo_lines.append(f"  - {ek_id}: {ek_desc}")

		skills_str = ",".join(ssc) if ssc else "-"
		bi_str = ",".join(bis) if bis else "-"

		topic_blocks.append(
			safe_join_lines([
				f"T{topic_id} {topic_name} | skills:{skills_str} | big:{bi_str}",
				*lo_lines
			])
		)

	# -----------------------
	# Course-level big ideas (VAR/UNC/DAT) full descriptions (NO truncation)
	# -----------------------
	if include_course_big_ideas and used_course_big_ids:
		for bi in course_spec.get("big_ideas", []) or []:
			bi_id = str(bi.get("id", "")).strip()
			if not bi_id or bi_id not in used_course_big_ids:
				continue
			course_big_name[bi_id] = normalize_whitespace(bi.get("name", "") or "")
			course_big_desc[bi_id] = normalize_whitespace(bi.get("description", "") or "")

	# -----------------------
	# Skill descriptions (compact but readable)
	# -----------------------
	skill_desc_lines: List[str] = []
	if include_skill_descriptions and used_skill_codes:
		for code in sorted(used_skill_codes):
			desc = trunc(skill_lookup.get(code, {}).get("description", ""), max_skill_desc_chars)
			skill_desc_lines.append(f"{code}={desc}" if desc else code)

	# -----------------------
	# Topic big idea descriptions (VAR-1 etc.) — short, useful
	# If some are missing from topic_big_desc, fall back to big_idea_lookup
	# -----------------------
	topic_big_desc_lines: List[str] = []
	if include_topic_big_idea_descriptions and used_topic_big_ids:
		for bi_id in sorted(used_topic_big_ids):
			desc = topic_big_desc.get(bi_id, "")
			if not desc:
				# fallback: your big_idea_lookup may store these if you built it that way
				desc = normalize_whitespace(big_idea_lookup.get(bi_id, {}).get("description", "") or "")
			if desc:
				topic_big_desc_lines.append(f"{bi_id}: {desc}")
			else:
				topic_big_desc_lines.append(f"{bi_id}")

	# -----------------------
	# Build final unit context (with section line breaks)
	# -----------------------
	unit_context_parts: List[str] = [
		f"{course_spec.get('name', '').strip()} | Unit {unit_index + 1}: {unit.get('name', '').strip()}",
		"",
		"ALLOWED_SKILLS: " + ",".join(sorted(allowed_skill_codes)),
	]
	if skill_desc_lines:
		unit_context_parts.append("ALLOWED_SKILL_DESC: " + " ; ".join(skill_desc_lines))

	unit_context_parts.extend([
		"",
		"ALLOWED_LOS: " + ",".join(sorted(allowed_lo_ids)),
	])

	# Course big ideas
	if include_course_big_ideas and course_big_desc:
		unit_context_parts.append("")
		unit_context_parts.append("BIG_IDEAS: " + ",".join(sorted(course_big_desc.keys())))
		unit_context_parts.append("BIG_IDEAS_DESC:")
		for bi_id in sorted(course_big_desc.keys()):
			name = course_big_name.get(bi_id, "").strip()
			desc = (course_big_desc.get(bi_id, "") or "").strip()
			# Keep FULL (no truncation), but still single line per item
			if name:
				unit_context_parts.append(f"{bi_id} ({name}): {desc}")
			else:
				unit_context_parts.append(f"{bi_id}: {desc}")

	# Topic big ideas (VAR-1 etc.)
	if topic_big_desc_lines:
		unit_context_parts.append("")
		unit_context_parts.append("TOPIC_BIG_IDEAS: " + ",".join(sorted(used_topic_big_ids)))
		unit_context_parts.append("TOPIC_BIG_IDEA_DESC:")
		unit_context_parts.extend(topic_big_desc_lines)

	# Topics and LOs
	unit_context_parts.append("")
	unit_context_parts.append("TOPICS_AND_LOS:")
	unit_context_parts.extend(topic_blocks)

	unit_context = safe_join_lines(unit_context_parts)

	constraints = {
		"allowed_skill_codes": sorted(allowed_skill_codes),
		"allowed_lo_ids": sorted(allowed_lo_ids),
	}

	return unit_context, constraints



# -----------------------
# Gemini call
# -----------------------
def generate_repair_tsv(
	client,
	course_name: str,
	repair_prompt_template: Template,
	unit_context: str,
	missing: int,
	used_ids: List[str],
	error_summary: str,
	context_label: str
) -> str:
	prompt = repair_prompt_template.render(
		num_questions=missing,
		course_name=course_name,
		unit_context=unit_context,
		used_ids=",".join(used_ids),
		error_summary=error_summary
	)

	# log_block(f"[{context_label}] REPAIR PROMPT", prompt, MAX_PROMPT_CHARS)

	try:
		resp = client.models.generate_content(
			model=MODEL,
			contents=prompt
		)
	except Exception as e:
		log(f"[{context_label}] Repair Gemini call failed: {repr(e)}")
		return ""

	return (resp.text or "").strip()


def generate_mcq_tsv(client, course_name, prompt_template, unit_context, context_label: str) -> str:
	print("Generating MCQ Prompt...")
	prompt = prompt_template.render(
		num_questions=QUESTIONS_PER_SET,
		unit_context=unit_context,
		course_name=course_name
	)

	# if PRINT_PROMPT:
	# 	log_block(f"[{context_label}] PROMPT", prompt, MAX_PROMPT_CHARS)

	# print(prompt)
	# print("\n")

	if len(prompt) > MAX_PROMPT_CHARS:
		print(f"[{context_label}] WARNING: Prompt length {len(prompt)} exceeds {MAX_PROMPT_CHARS}")

	print("Generating MCQ TSV...")
	try:
		resp = client.models.generate_content(
			model=MODEL,
			contents=prompt
		)
	except Exception as e:
		log(f"[{context_label}] Gemini call failed: {repr(e)}")
		return ""

	out = (resp.text or "").strip()

	# if PRINT_TSV:
	# 	log_block(f"[{context_label}] TSV OUTPUT", out, MAX_TSV_CHARS)

	return out


# -----------------------
# TSV parsing
# -----------------------
def parse_tsv(tsv_text: str, context_label: str) -> List[List[str]]:
	lines = [ln for ln in (tsv_text or "").splitlines() if ln.strip()]
	log(f"[{context_label}] parse_tsv: {len(lines)} non-empty lines found")

	rows = []
	for i, ln in enumerate(lines, start=1):
		cols = ln.split("\t")

		# Auto-fix: missing stimulus_payload column
		if len(cols) == 11:
			cols.append("")

		rows.append(cols)

	return rows



def tsv_to_questions(rows: List[List[str]], constraints: Dict[str, Any], context_label: str) -> List[dict]:
	questions = []
	rejected = {
		"row_invalid": 0,
		"skill_not_allowed": 0,
		"lo_not_allowed": 0,
		"svg_invalid": 0,
		"table_invalid": 0
	}

	for row_i, cols in enumerate(rows, start=1):
		error = validate_tsv_row(cols)
		if error:
			rejected["row_invalid"] += 1
			log(f"[{context_label}] Row {row_i} rejected: {error} | cols={len(cols)}")
			# if DEBUG:
			# 	print(cols)
			continue

		qid, diff, skills, los, idx, qtext, A, B, C, D, stim_type, stim_payload = [
			c.strip() for c in cols
		]

		correct_index = int(idx)

		skill_codes = [s.strip() for s in skills.split(",") if s.strip()]
		if any(s not in constraints["allowed_skill_codes"] for s in skill_codes):
			rejected["skill_not_allowed"] += 1
			log(f"[{context_label}] Row {row_i} rejected: skill_codes not allowed | {skill_codes}")
			continue

		lo_ids = [s.strip() for s in los.split(",") if s.strip()]
		if any(lo not in constraints["allowed_lo_ids"] for lo in lo_ids):
			rejected["lo_not_allowed"] += 1
			log(f"[{context_label}] Row {row_i} rejected: LO ids not allowed | {lo_ids}")
			continue

		stimulus = None
		if stim_type == "svg":
			if not is_valid_svg(stim_payload):
				rejected["svg_invalid"] += 1
				log(f"[{context_label}] Row {row_i} rejected: invalid svg payload")
				continue
			stimulus = {"type": "svg", "content": stim_payload}

		elif stim_type == "table":
			html_table = pipe_table_to_html(stim_payload)
			if "<table" not in html_table:
				rejected["table_invalid"] += 1
				log(f"[{context_label}] Row {row_i} rejected: table could not be converted to html table")
				continue
			stimulus = {"type": "table", "content": html_table}

		questions.append({
			"id": qid,
			"difficulty": diff,
			"skill_codes": skill_codes,
			"aligned_lo_ids": lo_ids,
			"correct_choice_index": correct_index,
			"question": qtext,
			"choices": [A, B, C, D],
			"stimulus": stimulus
		})

	log(f"[{context_label}] tsv_to_questions: accepted={len(questions)} rejected={rejected}")
	return questions


# -----------------------
# Render + save
# -----------------------
def render_html(html_template, course, unit_title, unit_index, set_index, questions, context_label: str):
	out_dir = OUTPUT_DIR / course / "mcq"
	ensure_dir(out_dir)

	html = html_template.render(
		course=course,
		unit=f"Unit {unit_index + 1}: {unit_title}",
		questions=questions
	)

	path = out_dir / f"unit{unit_index + 1}-set{set_index + 1}.html"

	if path.exists():
		log(f"[{context_label}] Skipping existing file: {path}")
		return path

	path.write_text(html, encoding="utf-8")
	log(f"[{context_label}] Wrote file: {path}")
	return path


# -----------------------
# Main
# -----------------------
def main():
	log("Starting MCQ generation run")

	client = init_client()
	prompt_template = Template(load_text(PROMPT_PATH))
	repair_prompt_template = Template(load_text(Path("utils/prompts/repair_prompt.txt")))
	html_template = Template(load_text(HTML_TEMPLATE_PATH))

	for course in AP_COURSES:
		course_spec = load_json(CONTENT_DIR / f"{course}.json")
		skill_lookup = build_skill_lookup(course_spec)
		big_idea_lookup = build_big_idea_lookup(course_spec)

		for set_index in range(NUM_SETS_PER_UNIT):
			for unit_index, unit in enumerate(course_spec.get("units", [])):
				unit_context, constraints = build_unit_context(
					course_spec,
					unit,
					unit_index,
					skill_lookup,
					big_idea_lookup
				)

				context_label = log_context(course, unit_index, unit.get("name", ""), set_index)
				log(f"[{context_label}] Generating set")

				all_questions: List[dict] = []
				used_ids: set = set()

				# ---------- INITIAL GENERATION ----------
				tsv = generate_mcq_tsv(
					client,
					course,
					prompt_template,
					unit_context,
					context_label
				)

				rows = parse_tsv(tsv, context_label)
				valid, invalid = validate_rows_individually(rows, constraints, context_label)

				all_questions.extend(valid)
				used_ids.update(q["id"] for q in valid)

				# ---------- REPAIR LOOP ----------
				repair_round = 0
				while len(all_questions) < QUESTIONS_PER_SET and repair_round < MAX_RETRIES_PER_SET:
					repair_round += 1
					missing = QUESTIONS_PER_SET - len(all_questions)

					log(f"[{context_label}] Repair round {repair_round}, missing={missing}")

					error_summary = summarize_errors(invalid)

					repair_tsv = generate_repair_tsv(
						client,
						course,
						repair_prompt_template,
						unit_context,
						missing,
						list(used_ids),
						error_summary,
						context_label
					)

					if not repair_tsv:
						break

					repair_rows = parse_tsv(repair_tsv, context_label)
					valid, invalid = validate_rows_individually(
						repair_rows, constraints, context_label
					)

					for q in valid:
						if q["id"] not in used_ids:
							all_questions.append(q)
							used_ids.add(q["id"])

				# ---------- FINAL CHECK ----------
				if len(all_questions) != QUESTIONS_PER_SET:
					log(f"[{context_label}] ❌ Failed to reach {QUESTIONS_PER_SET}")
					continue

				path = render_html(
					html_template,
					course,
					unit.get("name", ""),
					unit_index,
					set_index,
					all_questions,
					context_label
				)

				if path:
					log(f"[{context_label}] ✅ SUCCESS")
					break
				break
			break
		break


if __name__ == "__main__":
	main()
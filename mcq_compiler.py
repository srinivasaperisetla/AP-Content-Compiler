'''
Problems right now:
- Find a way to tag each question with easy hard medium etc.
- Make sure prompts work across all AP course for stem atleast for stem. 
- Comment the code properly and neatly
- Find a way to make sure that questions are not repeating across sets and within sets dedupe logic. 
- Add new repair prompt to feed context into invalid questions

- Optional Split generation into: planning pass (topics → question plan) then generation pass (plan → TSV)
- Optional Parallelize of units
'''

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import asyncio

from dotenv import load_dotenv
from google import genai
from jinja2 import Template
import certifi

from utility_functions import (
	log,
	log_block,
	log_context,
	init_client,
	load_json,
	load_text,
	ensure_dir,
	safe_join_lines,
	normalize_whitespace,
	build_skill_lookup,
	build_big_idea_lookup,
	summarize_invalid_reports,
	compress_lo_description,
	initialize_lo_coverage,
	get_priority_los,
)

from utils.image_generator import (
	generate_image_from_prompt,
	create_image_filename,
	enhance_prompt_for_image_generation
)

from utils.batch_state_manager import (
	BatchJobState,
	SetData,
	save_batch_job_state
)

from utils.batch_image_generator import (
	create_batch_job_for_unit
)

#Environment and Variables
load_dotenv()  # Load environment variables from .env file
os.environ["SSL_CERT_FILE"] = certifi.where()

MODEL = "gemini-2.5-flash"

NUM_SETS_PER_UNIT = 20
QUESTIONS_PER_SET = 25
MAX_RETRIES_PER_SET = 4

AP_COURSES = {
	# "AP Statistics": "ap_statistics",
	# "AP Physics 1": "ap_physics_1",
	# "AP Chemistry": "ap_chemistry",
	"AP Environmental Science": "ap_environmental_science",
}

CONTENT_DIR = Path("utils/content")
PROMPT_PATH = Path("utils/prompts/mcq_prompt.txt")
REPAIR_PROMPT_PATH = Path("utils/prompts/mcq_repair_prompt.txt")
HTML_TEMPLATE_PATH = Path("utils/templates/mcq.html")
OUTPUT_DIR = Path("output")

DEBUG = True
MAX_PROMPT_CHARS = 6000


# Validation Functions
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
			if DEBUG:
				log(f"[{context_label}] Row {row_i} rejected: {error} | cols={len(cols)}")
			continue

		diff, skills, los, idx, qtext, A, B, C, D, stim_type, stim_payload = cols = [
			c.strip() for c in cols
		]

		skill_codes = [s for s in skills.split(",") if s]
		if any(s not in constraints["allowed_skill_codes"] for s in skill_codes):
			invalid_reports.append({
				"row_index": row_i,
				"reason": "skill_not_allowed",
				"detail": skills
			})
			if DEBUG:
				invalid_skills = [s for s in skill_codes if s not in constraints["allowed_skill_codes"]]
				log(f"[{context_label}] Row {row_i} rejected: Invalid skills {invalid_skills}")
			continue

		lo_ids = [s for s in los.split(",") if s]
		if any(lo not in constraints["allowed_lo_ids"] for lo in lo_ids):
			invalid_reports.append({
				"row_index": row_i,
				"reason": "lo_not_allowed",
				"detail": los
			})
			if DEBUG:
				invalid_los = [lo for lo in lo_ids if lo not in constraints["allowed_lo_ids"]]
				log(f"[{context_label}] Row {row_i} rejected: Invalid LOs {invalid_los}")
			continue

		if stim_type == "image":
			# Validation for image prompt format
			if not stim_payload.startswith("IMAGE_PROMPT:"):
				invalid_reports.append({
					"row_index": row_i,
					"reason": "image_prompt_missing",
					"detail": "Image stimulus must start with 'IMAGE_PROMPT:'"
				})
				if DEBUG:
					log(f"[{context_label}] Row {row_i} rejected: Missing IMAGE_PROMPT: prefix")
				continue

			# Extract and validate prompt length
			prompt = stim_payload.replace("IMAGE_PROMPT:", "").strip()
			if len(prompt) < 20:
				invalid_reports.append({
					"row_index": row_i,
					"reason": "image_prompt_too_short",
					"detail": "Image prompt must be detailed (min 20 chars)"
				})
				if DEBUG:
					log(f"[{context_label}] Row {row_i} rejected: Image prompt too short")
				continue

		valid_questions.append({
		"id": None,
			"difficulty": diff,
			"skill_codes": skill_codes,
			"aligned_lo_ids": lo_ids,
			"correct_choice_index": int(idx),
			"question": qtext,
			"choices": [A, B, C, D],
		"stimulus_type": stim_type,
		"stimulus_content": stim_payload,
		"stimulus": None  # Will be populated during image generation
	})


	return valid_questions, invalid_reports

def validate_tsv_row(cols: List[str]) -> Optional[str]:
	if len(cols) != 11:
		return "Wrong column count"

	diff, skills, los, idx, qtext, A, B, C, D, stim_type, stim_payload = cols

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

	if stim_type not in {"none", "image"}:
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

	if stim_type == "image" and not stim_payload.startswith("IMAGE_PROMPT:"):
		return "Image stimulus must start with 'IMAGE_PROMPT:'"
	
	if stim_type == "image":
		prompt = stim_payload.replace("IMAGE_PROMPT:", "").strip()
		if len(prompt) < 20:
			return "Image prompt must be detailed (min 20 chars)"

	return None


# Unit context builder
def build_unit_context(
	course_spec: dict,
	unit: dict,
	unit_index: int,
	skill_lookup: Dict[str, Dict[str, str]],
	big_idea_lookup: Dict[str, Dict[str, str]],
	question_type: str = "mcq",  # NEW: "mcq" or "frq"
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
			# Apply LO compression before truncation
			lo_desc_raw = lo.get("description", "")
			lo_desc_compressed = compress_lo_description(lo_desc_raw)
			lo_desc = trunc(lo_desc_compressed, max_lo_desc_chars)
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
	# Exam Section Context (NEW)
	# -----------------------
	exam_context_lines = []
	section_key = "I" if question_type == "mcq" else "II"
	
	for section in course_spec.get("exam_sections", []):
		if section.get("section") == section_key:
			# Only include descriptions array
			descriptions = section.get("descriptions", [])
			if descriptions:
				exam_context_lines.append("EXAM_CONTEXT:")
				for desc in descriptions:
					# Keep full descriptions, no truncation
					exam_context_lines.append(f"- {desc}")
			
			break  # Only one section
	
	# -----------------------
	# Task Verbs (FRQ only) (NEW)
	# -----------------------
	task_verb_lines = []
	if question_type == "frq":
		# Prioritized verbs for AP Statistics
		priority_verbs = [
			"Calculate", "Explain", "Justify", "Describe",
			"Interpret", "Compare", "Identify", "Construct",
			"Determine", "Verify"
		]
		
		task_verb_lines.append("TASK_VERBS:")
		for verb_obj in course_spec.get("task_verbs", []):
			verb = verb_obj.get("verb", "")
			# Check if this is a priority verb
			if any(pv in verb for pv in priority_verbs):
				desc = verb_obj.get("description", "")
				# Compress description (max 100 chars)
				desc_short = desc[:100] + ("..." if len(desc) > 100 else "")
				task_verb_lines.append(f"  {verb}: {desc_short}")

	# -----------------------
	# Build final unit context (with section line breaks)
	# -----------------------
	unit_context_parts: List[str] = [
		f"{course_spec.get('name', '').strip()} | Unit {unit_index + 1}: {unit.get('name', '').strip()}",
		"",
	]
	
	# Add exam context early (NEW)
	if exam_context_lines:
		unit_context_parts.extend(exam_context_lines)
		unit_context_parts.append("")
	
	# Continue with existing
	unit_context_parts.append("ALLOWED_SKILLS: " + ",".join(sorted(allowed_skill_codes)))
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
	
	# Task verbs at end (FRQ only) (NEW)
	if task_verb_lines:
		unit_context_parts.append("")
		unit_context_parts.extend(task_verb_lines)

	unit_context = safe_join_lines(unit_context_parts)

	constraints = {
		"allowed_skill_codes": sorted(allowed_skill_codes),
		"allowed_lo_ids": sorted(allowed_lo_ids),
	}

	return unit_context, constraints


async def process_set_for_batch(
	text_sem: asyncio.Semaphore,
	client: genai.Client,
	course_name: str,
	course_id: str,
	unit: dict,
	unit_index: int,
	set_index: int,
	prompt_template: Template,
	repair_prompt_template: Template,
	html_template: Template,
	unit_context: str,
	constraints: dict,
	coverage_tracker: Dict[str, int]
) -> Optional[Dict]:
	"""
	Process a single set in batch mode - generate text, collect image prompts.
	
	Returns:
		Dict with set_index, questions_data, image_requests, or None if file exists
	"""
	context_label = f"{course_id} | U{unit_index+1} | Set{set_index+1}"
	unit_title = unit.get("name", "")
	
	# CHECK IF FILE EXISTS FIRST
	out_dir = OUTPUT_DIR / course_id / "mcq"
	ensure_dir(out_dir)
	output_path = out_dir / f"unit{unit_index + 1}-set{set_index + 1}.html"
	
	if output_path.exists():
		log(f"[{context_label}] ✓ File already exists, skipping")
		return None
	
	# Generate text questions (same as real-time mode)
	async with text_sem:
		log(f"[{context_label}] Generating questions...")
		all_questions = []
		
		# Get under-covered LOs
		priority_los = get_priority_los(
			coverage_tracker,
			constraints["allowed_lo_ids"],
			top_n=10
		)
		priority_los_str = ",".join(priority_los)
		
		prompt = prompt_template.render(
			num_questions=QUESTIONS_PER_SET,
			unit_context=unit_context,
			course_name=course_name,
			priority_los=priority_los_str
		)
		
		try:
			response = await client.aio.models.generate_content(
				model=MODEL,
				contents=prompt
			)
			tsv = response.text or ""
		except Exception as e:
			log(f"[{context_label}] Initial API Error: {e}")
			tsv = ""
		
		rows = parse_tsv(tsv, context_label)
		valid, invalid_initial = validate_rows_individually(rows, constraints, context_label)
		all_questions.extend(valid)
		all_invalid_reports = invalid_initial.copy()
		
		# Repair loop (same as real-time mode)
		repair_round = 0
		while len(all_questions) < QUESTIONS_PER_SET and repair_round < MAX_RETRIES_PER_SET:
			repair_round += 1
			missing = QUESTIONS_PER_SET - len(all_questions)
			
			if missing <= 3:
				request_count = missing + 3
			else:
				buffer = max(2, min(10, int(missing * 0.5)))
				request_count = missing + buffer
			
			repair_summary = summarize_invalid_reports(all_invalid_reports)
			repair_prompt = repair_prompt_template.render(
				num_questions=request_count,
				unit_context=unit_context,
				course_name=course_name,
				validation_errors=repair_summary,
				priority_los=priority_los_str
			)
			
			try:
				repair_response = await client.aio.models.generate_content(
					model=MODEL,
					contents=repair_prompt
				)
				repair_tsv = repair_response.text or ""
			except Exception as e:
				log(f"[{context_label}] Repair API Error: {e}")
				break
			
			repair_rows = parse_tsv(repair_tsv, context_label)
			repair_valid, repair_invalid = validate_rows_individually(repair_rows, constraints, context_label)
			all_questions.extend(repair_valid)
			all_invalid_reports.extend(repair_invalid)
	
	# Check if we have enough questions
	if len(all_questions) < QUESTIONS_PER_SET:
		log(f"[{context_label}] ❌ FAILED: Only got {len(all_questions)}/{QUESTIONS_PER_SET}")
		return None
	
	all_questions = all_questions[:QUESTIONS_PER_SET]
	
	# Collect image prompts
	image_requests = []
	for q_idx, question in enumerate(all_questions):
		if question.get("stimulus_type") == "image":
			content = question.get("stimulus_content", "")
			
			if content.startswith("IMAGE_PROMPT:"):
				raw_prompt = content.replace("IMAGE_PROMPT:", "").strip()
				
				# Enhance prompt
				question_stem = question.get("question", "")
				choices = question.get("choices", ["", "", "", ""])
				answer_choices = "\n".join([
					f"A. {choices[0]}",
					f"B. {choices[1]}",
					f"C. {choices[2]}",
					f"D. {choices[3]}"
				])
				enhanced_prompt = enhance_prompt_for_image_generation(
					raw_prompt=raw_prompt,
					question_stem=question_stem,
					question_type="MCQ",
					course_name=course_name,
					answer_choices=answer_choices,
					correct_answer_index=question.get("correct_choice_index", -1)
				)
				
				# Key format: u{unit}_s{set}_q{question}
				image_requests.append({
					"key": f"u{unit_index+1}_s{set_index+1}_q{q_idx}",
					"prompt": enhanced_prompt,
					"question_index": q_idx
				})
	
	# If no images needed, render HTML immediately
	if len(image_requests) == 0:
		assign_question_ids(all_questions, course_id, unit_index, set_index)
		render_html(
			html_template=html_template,
			course_name=course_name,
			course_id=course_id,
			unit_title=unit_title,
			unit_index=unit_index,
			set_index=set_index,
			questions=all_questions,
			context_label=context_label
		)
		log(f"[{context_label}] ✓ No images needed, HTML rendered")
		
		# Update coverage tracker
		for question in all_questions:
			for lo_id in question.get("aligned_lo_ids", []):
				if lo_id in coverage_tracker:
					coverage_tracker[lo_id] += 1
		
		return None  # Don't include in batch
	
	log(f"[{context_label}] ✓ Generated {len(all_questions)} questions, {len(image_requests)} images needed")
	
	# Return data for batch processing
	return {
		"set_index": set_index,
		"questions_data": all_questions,
		"image_requests": image_requests
	}


# Gemini call to process a single set. 
async def process_single_set(
	text_sem: asyncio.Semaphore,
	image_sem: asyncio.Semaphore,
	client: genai.Client,
	course_name: str,
	course_id: str,
	unit: dict, 
	unit_index: int, 
	set_index: int,
	prompt_template: Template,
	repair_prompt_template: Template,
	html_template: Template,
	unit_context: str,
	constraints: dict,
	coverage_tracker: Dict[str, int]
):
	context_label = f"{course_id} | U{unit_index+1} | Set{set_index+1}"
	unit_title = unit.get("name", "")
	
	# CHECK IF FILE EXISTS FIRST - EXIT EARLY TO SAVE API QUOTA
	out_dir = OUTPUT_DIR / course_id / "mcq"
	ensure_dir(out_dir)
	output_path = out_dir / f"unit{unit_index + 1}-set{set_index + 1}.html"
	
	if output_path.exists():
		log(f"[{context_label}] ✓ File already exists, skipping generation: {output_path}")
		return output_path
	
	# Wait for permission from Text Semaphore (Rate Limit Guard)
	async with text_sem:
		log(f"[{context_label}] Starting generation...")
		all_questions = []
		
		# ----------------------------------------
		# 1. Initial Generation
		# ----------------------------------------
		
		# Get under-covered LOs
		priority_los = get_priority_los(
			coverage_tracker,
			constraints["allowed_lo_ids"],
			top_n=10  # Top 10 least-covered
		)
		priority_los_str = ",".join(priority_los)
		
		prompt = prompt_template.render(
			num_questions=QUESTIONS_PER_SET,
			unit_context=unit_context,
			course_name=course_name,
			priority_los=priority_los_str
		)
		
		try:
			# ASYNC CALL
			response = await client.aio.models.generate_content(
				model=MODEL,
				contents=prompt
			)
			tsv = response.text or ""
		except Exception as e:
			log(f"[{context_label}] Initial API Error: {e}")
			tsv = ""
		
		# Synchronous Parsing
		rows = parse_tsv(tsv, context_label)
		valid, invalid_initial = validate_rows_individually(rows, constraints, context_label)
		all_questions.extend(valid)
		
		# Track all invalid reports for error summary
		all_invalid_reports = invalid_initial.copy()
		
		# ----------------------------------------
		# 2. Repair Loop
		# ----------------------------------------
		repair_round = 0
		while len(all_questions) < QUESTIONS_PER_SET and repair_round < MAX_RETRIES_PER_SET:
			repair_round += 1
			missing = QUESTIONS_PER_SET - len(all_questions)
			
			# Add buffer to repair requests (ask for more than needed)
			if missing <= 3:
				# For small requests, add fixed buffer of 3
				request_count = missing + 3
			else:
				# For larger requests, add 30-50% buffer (min 2, max 10)
				buffer = max(2, min(10, int(missing * 0.5)))
				request_count = missing + buffer
			
			log(f"[{context_label}] Repair {repair_round}: missing {missing}, requesting {request_count}")
			
			# Generate error summary from accumulated invalid reports
			error_summary = summarize_invalid_reports(all_invalid_reports)
			
			# Preview of allowed constraints (first 10)
			allowed_skills_preview = ",".join(constraints["allowed_skill_codes"][:10])
			allowed_los_preview = ",".join(constraints["allowed_lo_ids"][:10])
			
			repair_prompt_text = repair_prompt_template.render(
				num_questions=request_count,
		unit_context=unit_context,
				course_name=course_name,
				error_summary=error_summary,
				allowed_skills_preview=allowed_skills_preview,
				allowed_los_preview=allowed_los_preview
			)
			
			try:
				repair_resp = await client.aio.models.generate_content(
					model=MODEL,
					contents=repair_prompt_text
				)
				repair_tsv = repair_resp.text or ""
				
				repair_rows = parse_tsv(repair_tsv, context_label)
				valid_repair, invalid_repair = validate_rows_individually(
					repair_rows,
					constraints,
					context_label
				)
				all_questions.extend(valid_repair)
				all_invalid_reports.extend(invalid_repair)  # Accumulate for next repair
			
			except Exception as e:
				log(f"[{context_label}] Repair API Error: {e}")
				break
		
	# ----------------------------------------
	# 3. Final Check & Save
	# ----------------------------------------
	if len(all_questions) >= QUESTIONS_PER_SET:
		all_questions = all_questions[:QUESTIONS_PER_SET]
		
		# ----------------------------------------
		# 4. Generate Images for Stimuli
		# ----------------------------------------
		log(f"[{context_label}] Starting image generation for questions with stimulus_type='image'...")
		images_generated = 0
		images_failed = 0
		
		for q_idx, question in enumerate(all_questions):
			if question.get("stimulus_type") == "image":
				content = question.get("stimulus_content", "")
				
				if content.startswith("IMAGE_PROMPT:"):
					raw_prompt = content.replace("IMAGE_PROMPT:", "").strip()
					
					# Enhance prompt for better image generation with full context
					question_stem = question.get("question", "")
					# Build answer choices string for context
					choices = question.get("choices", ["", "", "", ""])
					answer_choices = "\n".join([
						f"A. {choices[0]}",
						f"B. {choices[1]}",
						f"C. {choices[2]}",
						f"D. {choices[3]}"
					])
					enhanced_prompt = enhance_prompt_for_image_generation(
						raw_prompt=raw_prompt,
						question_stem=question_stem,
						question_type="MCQ",
						course_name=course_name,
						answer_choices=answer_choices,
						correct_answer_index=question.get("correct_choice_index", -1)
					)
					
					# Create filename
					filename = create_image_filename(
						course_name=course_name,
						question_type="mcq",
						unit_id=unit["id"],
						set_index=set_index,
						question_index=q_idx
					)
					
				# Generate and save image (with image semaphore for rate limiting)
				image_path = OUTPUT_DIR / "images" / course_name / f"unit_{unit['id']}" / filename
				
				# Use IMAGE semaphore with rate limiting
				async with image_sem:
					# Enforce 3-second spacing between image requests
					await asyncio.sleep(3)
					
					success, base64_data = await generate_image_from_prompt(
						prompt=enhanced_prompt,
						output_path=image_path,
						api_key=os.getenv("GOOGLE_API_KEY"),
						aspect_ratio="1:1"  # Can adjust based on stimulus type if needed
					)
					
					if success:
						# Update question with image data
						question["stimulus"] = {
							"type": "image",
							"file_path": str(image_path),
							"base64": base64_data,
							"alt_text": raw_prompt  # Use original prompt as alt text
						}
						images_generated += 1
						log(f"[{context_label}] Generated image: {filename}")
					else:
						images_failed += 1
						log(f"[{context_label}] WARNING: Failed to generate image for Q{q_idx+1}")
						# Keep question but mark stimulus as failed
						question["stimulus"] = {
							"type": "none",
							"error": "Image generation failed"
						}
		
		log(f"[{context_label}] Image generation complete: {images_generated} success, {images_failed} failed")
		
		# Fail if any images didn't generate
		if images_failed > 0:
			log(f"[{context_label}] ❌ FAILED: {images_failed} image(s) failed to generate. Skipping HTML creation.")
			return None
		
		assign_question_ids(
			all_questions,
			course_id,
			unit_index,
			set_index
		)
		
		render_html(
			html_template=html_template,
			course_name=course_name,
			course_id=course_id,
			unit_title=unit_title,
			unit_index=unit_index,
			set_index=set_index,
			questions=all_questions,
			context_label=context_label
		)
		
		# Update coverage tracker
		for question in all_questions:
			for lo_id in question.get("aligned_lo_ids", []):
				if lo_id in coverage_tracker:
					coverage_tracker[lo_id] += 1
		
		log(f"[{context_label}] ✅ SUCCESS: Saved {len(all_questions)} questions")
	else:
		log(f"[{context_label}] ❌ FAILED: Only got {len(all_questions)}/{QUESTIONS_PER_SET}")

# TSV parsing
def parse_tsv(tsv_text: str, context_label: str) -> List[List[str]]:
	lines = [ln for ln in (tsv_text or "").splitlines() if ln.strip()]
	log(f"[{context_label}] parse_tsv: {len(lines)} non-empty lines found")

	rows = []
	for i, ln in enumerate(lines, start=1):
		cols = ln.split("\t")

		rows.append(cols)

	return rows


# Render + save to HTML Template
def render_html(html_template, course_name, course_id, unit_title, unit_index, set_index, questions, context_label: str):
	out_dir = OUTPUT_DIR / course_id / "mcq"
	ensure_dir(out_dir)

	html = html_template.render(
		course=course_name,
		unit=f"Unit {unit_index + 1}: {unit_title}",
		questions=questions
	)

	path = out_dir / f"unit{unit_index + 1}-set{set_index + 1}.html"
	path.write_text(html, encoding="utf-8")
	log(f"[{context_label}] Wrote file: {path}")
	return path

def assign_question_ids(
	questions: List[dict],
	course_id: str,
	unit_index: int,
	set_index: int
) -> None:
	for i, q in enumerate(questions, start=1):
		q["id"] = f"{course_id}_MCQ_U{unit_index + 1}S{set_index + 1}Q{i}"



# Main Async Process
async def main_async(batch_mode: bool = False, target_course: Optional[str] = None):
	"""
	Main async process for MCQ generation.
	
	Args:
		batch_mode: If True, use batch API for image generation (two-phase)
		target_course: If provided, only process this course (e.g., 'ap_physics_1')
	"""
	client = init_client()
	
	# Load templates ONCE
	prompt_template = Template(load_text(PROMPT_PATH))
	repair_prompt_template = Template(load_text(REPAIR_PROMPT_PATH))
	html_template = Template(load_text(HTML_TEMPLATE_PATH))

	# TWO semaphores for different rate limits (not used in batch mode for images)
	text_sem = asyncio.Semaphore(60)  # High parallelism for text generation
	image_sem = asyncio.Semaphore(5)  # Strict serialization for image generation (real-time mode only)
	
	# Filter courses if target specified
	courses_to_process = {}
	if target_course:
		if target_course in AP_COURSES:
			courses_to_process = {AP_COURSES[target_course]: target_course}
		else:
			print(f"Error: Course '{target_course}' not found in AP_COURSES")
			return
	else:
		courses_to_process = {v: k for k, v in AP_COURSES.items()}
	
	if batch_mode:
		log("=" * 60)
		log("BATCH MODE: Phase 1 - Text Generation & Batch Submission")
		log("=" * 60)
		await run_batch_mode(
			client,
			courses_to_process,
			prompt_template,
			repair_prompt_template,
			html_template,
			text_sem
		)
	else:
		log("=" * 60)
		log("REAL-TIME MODE: Text & Image Generation")
		log("=" * 60)
		await run_realtime_mode(
			client,
			courses_to_process,
			prompt_template,
			repair_prompt_template,
			html_template,
			text_sem,
			image_sem
		)


async def run_realtime_mode(
	client,
	courses_to_process,
	prompt_template,
	repair_prompt_template,
	html_template,
	text_sem,
	image_sem
):
	"""Run in real-time mode (original behavior)."""
	tasks = []
	
	for course_id, course_name in courses_to_process.items():
		course_spec = load_json(CONTENT_DIR / f"{course_id}.json")
		skill_lookup = build_skill_lookup(course_spec)
		big_idea_lookup = build_big_idea_lookup(course_spec)

		for unit_index, unit in enumerate(course_spec.get("units", [])):
			# Initialize coverage tracker per unit
			coverage_tracker = initialize_lo_coverage(unit)

			unit_context, constraints = build_unit_context(
				course_spec,
				unit,
				unit_index,
				skill_lookup,
				big_idea_lookup,
				question_type="mcq"
			)
			
			for set_index in range(NUM_SETS_PER_UNIT):
				tasks.append(
					process_single_set(
						text_sem,
						image_sem,
						client,
						course_name,
						course_id,
						unit,
						unit_index,
						set_index,
						prompt_template,
						repair_prompt_template,
						html_template,
						unit_context,
						constraints,
						coverage_tracker
					)
				)
	
	print(f"Starting {len(tasks)} parallel tasks...")
	await asyncio.gather(*tasks)


async def process_unit_for_batch(
	client,
	course_id,
	course_name,
	unit,
	unit_index,
	skill_lookup,
	big_idea_lookup,
	prompt_template,
	repair_prompt_template,
	html_template,
	text_sem
):
	"""
	Process a single unit in batch mode - generate text in parallel, then create batch job.
	
	Returns:
		Tuple of (batch_jobs_created, total_images) for this unit
	"""
	log(f"\n{'='*60}")
	log(f"Processing {course_name} - Unit {unit_index + 1}: {unit.get('name', '')}")
	log(f"{'='*60}")
	
	# Initialize coverage tracker per unit
	coverage_tracker = initialize_lo_coverage(unit)

	unit_context, constraints = build_unit_context(
		load_json(CONTENT_DIR / f"{course_id}.json"),
		unit,
		unit_index,
		skill_lookup,
		big_idea_lookup,
		question_type="mcq"
	)
	
	# Collect tasks for parallel text generation (one task per set)
	set_tasks = []
	for set_index in range(NUM_SETS_PER_UNIT):
		set_tasks.append(
			process_set_for_batch(
				text_sem,
				client,
				course_name,
				course_id,
				unit,
				unit_index,
				set_index,
				prompt_template,
				repair_prompt_template,
				html_template,
				unit_context,
				constraints,
				coverage_tracker
			)
		)
	
	# Execute all set tasks in parallel
	log(f"[Unit {unit_index+1}] Generating text for {len(set_tasks)} sets in parallel...")
	set_results = await asyncio.gather(*set_tasks)
	
	# Process results and collect image requests
	all_sets_data = []
	all_image_requests = []
	sets_with_images = 0
	
	for set_data in set_results:
		if set_data:
			all_sets_data.append(set_data)
			all_image_requests.extend(set_data["image_requests"])
			if len(set_data["image_requests"]) > 0:
				sets_with_images += 1
	
	# Create batch job for this unit if there are images
	if len(all_image_requests) > 0:
		unit_label = f"{course_id}_u{unit_index+1}_mcq"
		
		log(f"\n[Unit {unit_index+1}] Creating batch job:")
		log(f"  - Sets with images: {sets_with_images}/{len(all_sets_data)}")
		log(f"  - Total images: {len(all_image_requests)}")
		
		try:
			job_name, uploaded_file, jsonl_path = await create_batch_job_for_unit(
				client,
				all_image_requests,
				unit_label
			)
			
			# Save state
			state = BatchJobState(
				job_name=job_name,
				course_id=course_id,
				course_name=course_name,
				unit_index=unit_index,
				unit_title=unit.get("name", ""),
				question_type="mcq",
				sets=[{
					"set_index": sd["set_index"],
					"questions_data": sd["questions_data"],
					"image_requests": sd["image_requests"]
				} for sd in all_sets_data if len(sd["image_requests"]) > 0],
				total_image_requests=len(all_image_requests),
				jsonl_file_path=str(jsonl_path),
				uploaded_file_name=uploaded_file,
				created_at=datetime.now().isoformat()
			)
			
			state_file = save_batch_job_state(state, Path("batch_jobs/state"))
			
			log(f"  ✓ Batch job created: {job_name}")
			log(f"  ✓ State saved: {state_file}")
			
			return 1, len(all_image_requests)
			
		except Exception as e:
			log(f"  ✗ Error creating batch job: {e}")
			return 0, 0
	else:
		log(f"[Unit {unit_index+1}] No images needed - all HTML files rendered")
		return 0, 0


async def run_batch_mode(
	client,
	courses_to_process,
	prompt_template,
	repair_prompt_template,
	html_template,
	text_sem
):
	"""Run in batch mode - generate text in parallel, submit batch jobs in parallel."""
	batch_jobs_created = 0
	total_images = 0
	
	# Collect all unit tasks for parallel processing
	unit_tasks = []
	
	for course_id, course_name in courses_to_process.items():
		course_spec = load_json(CONTENT_DIR / f"{course_id}.json")
		skill_lookup = build_skill_lookup(course_spec)
		big_idea_lookup = build_big_idea_lookup(course_spec)

		for unit_index, unit in enumerate(course_spec.get("units", [])):
			# if unit_index != 0:
			# 	continue
			
			# Add task for this unit (will process sets in parallel, then submit batch job)
			unit_tasks.append(
				process_unit_for_batch(
					client,
					course_id,
					course_name,
					unit,
					unit_index,
					skill_lookup,
					big_idea_lookup,
					prompt_template,
					repair_prompt_template,
					html_template,
					text_sem
				)
			)
	
	# Execute all unit tasks in parallel (text generation + batch job submission)
	log(f"\n{'='*60}")
	log(f"Starting parallel processing: {len(unit_tasks)} unit(s)")
	log(f"{'='*60}")
	
	unit_results = await asyncio.gather(*unit_tasks)
	
	# Aggregate results
	for jobs, images in unit_results:
		batch_jobs_created += jobs
		total_images += images
	
	log(f"\n{'='*60}")
	log(f"PHASE 1 COMPLETE")
	log(f"{'='*60}")
	log(f"Batch jobs created: {batch_jobs_created}")
	log(f"Total images queued: {total_images}")
	log(f"\nNext steps:")
	log(f"1. Wait for batch jobs to complete (up to 24 hours)")
	log(f"2. Run: python batch_retrieve_and_render.py")


def main():
	"""Main entry point with argument parsing."""
	parser = argparse.ArgumentParser(
		description='Generate AP-style MCQ questions with optional batch image generation'
	)
	parser.add_argument(
		'--course',
		type=str,
		help='Specific course ID to process (e.g., ap_physics_1). If not provided, processes all courses.'
	)
	parser.add_argument(
		'--batch',
		action='store_true',
		help='Use batch API for image generation (two-phase workflow)'
	)
	
	args = parser.parse_args()
	
	asyncio.run(main_async(
		batch_mode=args.batch,
		target_course=args.course
	))


if __name__ == "__main__":
	main()
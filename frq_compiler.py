'''
FRQ Compiler - Async/Parallel Pipeline
Follows the same architecture as mcq_compiler.py:
- TSV-based generation and validation
- Async API calls with rate limiting
- Row-by-row validation with repair loops
- Parallel task execution across units/sets
'''

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

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

# Environment and Variables
load_dotenv()  # Load environment variables from .env file
os.environ["SSL_CERT_FILE"] = certifi.where()

MODEL = "gemini-2.5-flash"

NUM_SETS_PER_UNIT = 20
FRQS_PER_SET = 5
MAX_RETRIES_PER_SET = 4

AP_COURSES = {
	"AP Statistics": "ap_statistics",
}

CONTENT_DIR = Path("utils/content")
PROMPT_PATH = Path("utils/prompts/frq_prompt.txt")
REPAIR_PROMPT_PATH = Path("utils/prompts/frq_repair_prompt.txt")
HTML_TEMPLATE_PATH = Path("utils/templates/frq.html")
OUTPUT_DIR = Path("output")

DEBUG = True
MAX_PROMPT_CHARS = 6000


# Parsing Helper Functions
def parse_parts(parts_string: str) -> List[dict]:
	"""
	Parse pipe-separated parts string into structured format.
	Example: "a. Describe...|b. Calculate...|c. Justify..."
	Returns: [{"label": "a", "prompt": "Describe..."}, ...]
	"""
	if not parts_string or not parts_string.strip():
		return []
	
	parts = []
	segments = parts_string.split("|")
	
	for segment in segments:
		segment = segment.strip()
		if not segment:
			continue
		
		# Match pattern like "a. " or "(a) " at the start
		match = re.match(r'^([a-z])[.)]\s*(.+)', segment, re.IGNORECASE | re.DOTALL)
		if match:
			label = match.group(1).lower()
			prompt = match.group(2).strip()
			parts.append({"label": label, "prompt": prompt})
		else:
			# No label found, use sequential letters
			label = chr(ord('a') + len(parts))
			parts.append({"label": label, "prompt": segment})
	
	return parts


def parse_scoring_guidelines(guidelines_string: str) -> List[str]:
	"""
	Parse pipe-separated scoring guidelines into list.
	Example: "Part a (1pt): Description...|Part b (2pts): Calculation..."
	Returns: ["Part a (1pt): Description...", "Part b (2pts): Calculation..."]
	"""
	if not guidelines_string or not guidelines_string.strip():
		return []
	
	guidelines = []
	segments = guidelines_string.split("|")
	
	for segment in segments:
		segment = segment.strip()
		if segment:
			guidelines.append(segment)
	
	return guidelines


# Validation Functions
def validate_tsv_row(cols: List[str]) -> Optional[str]:
	"""Validate a single FRQ TSV row."""
	if len(cols) != 8:
		return f"Wrong column count (expected 8, got {len(cols)})"
	
	diff, skills, los, context, parts, guidelines, stim_type, stim_payload = cols
	
	if diff not in {"easy", "medium", "hard"}:
		return "Invalid difficulty"
	
	if not skills.strip():
		return "Empty skill_codes"
	
	if not los.strip():
		return "Empty learning_objective_ids"
	
	if not context.strip():
		return "Empty context"
	
	if not parts.strip():
		return "Empty parts"
	
	# Parts must contain at least one labeled part
	if not re.search(r'[a-z][.)]', parts, re.IGNORECASE):
		return "Parts must contain labeled sections (a., b., etc.)"
	
	if stim_type not in {"none", "image"}:
		return "Invalid stimulus_type"
	
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


def validate_rows_individually(
	rows: List[List[str]],
	constraints: Dict[str, Any],
	context_label: str
) -> Tuple[List[dict], List[dict]]:
	"""
	Validate FRQ TSV rows individually.
	Returns:
	  valid_frqs: List[frq_dict]
	  invalid_reports: List[{row_index, reason, detail}]
	"""
	valid_frqs = []
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
		
		diff, skills, los, context, parts_str, guidelines_str, stim_type, stim_payload = [
			c.strip() for c in cols
		]
		
		# Validate skill codes
		skill_codes = [s.strip() for s in skills.split(",") if s.strip()]
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
		
		# Validate learning objectives
		lo_ids = [s.strip() for s in los.split(",") if s.strip()]
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
		
		# Validate image prompt if present
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
		
		# Parse parts
		parts_list = parse_parts(parts_str)
		if not parts_list:
			invalid_reports.append({
				"row_index": row_i,
				"reason": "parts_parse_failed",
				"detail": "Could not parse parts string"
			})
			continue
		
		# Parse scoring guidelines
		scoring_guidelines = parse_scoring_guidelines(guidelines_str)
		
		valid_frqs.append({
			"id": None,  # Will be assigned later
			"difficulty": diff,
			"skill_codes": skill_codes,
			"aligned_lo_ids": lo_ids,
			"context": context,
			"parts": parts_list,
			"scoring_guidelines": scoring_guidelines,
			"stimulus_type": stim_type,
			"stimulus_content": stim_payload,
			"stimulus": None  # Will be populated during image generation
		})
	
	return valid_frqs, invalid_reports


# Unit context builder (reuse from mcq_compiler logic)
def build_unit_context(
	course_spec: dict,
	unit: dict,
	unit_index: int,
	skill_lookup: Dict[str, Dict[str, str]],
	big_idea_lookup: Dict[str, Dict[str, str]],
	question_type: str = "frq",  # NEW
	max_ek_per_lo: int = 0,
	include_skill_descriptions: bool = True,
	include_course_big_ideas: bool = True,
	include_topic_big_idea_descriptions: bool = True,
	max_topic_name_chars: int = 90,
	max_lo_desc_chars: int = 200,
	max_skill_desc_chars: int = 140,
) -> Tuple[str, Dict[str, Any]]:
	"""Build compressed unit context string and constraints dict."""
	
	def trunc(s: str, n: int) -> str:
		s = normalize_whitespace(s or "")
		return s if len(s) <= n else s[: n - 3] + "..."
	
	allowed_skill_codes: set = set()
	allowed_lo_ids: set = set()
	
	used_skill_codes: set = set()
	used_topic_big_ids: set = set()
	topic_big_desc: Dict[str, str] = {}
	
	used_course_big_ids: set = set()
	course_big_desc: Dict[str, str] = {}
	course_big_name: Dict[str, str] = {}
	
	topic_blocks: List[str] = []
	
	# Extract per-topic content
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
		
		# Topic-level big ideas
		bis: List[str] = []
		for bi in (topic.get("big_ideas") or []):
			bi_id = str(bi.get("id", "")).strip()
			bi_d = normalize_whitespace(bi.get("description", "") or "")
			if not bi_id:
				continue
			
			bis.append(bi_id)
			used_topic_big_ids.add(bi_id)
			
			if include_topic_big_idea_descriptions and bi_d and bi_id not in topic_big_desc:
				topic_big_desc[bi_id] = bi_d
			
			if "-" in bi_id:
				prefix = bi_id.split("-", 1)[0]
				if prefix:
					used_course_big_ids.add(prefix)
		
		# Learning objectives
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
			
			# Optional EKs
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
	
	# Course-level big ideas
	if include_course_big_ideas and used_course_big_ids:
		for bi in course_spec.get("big_ideas", []) or []:
			bi_id = str(bi.get("id", "")).strip()
			if not bi_id or bi_id not in used_course_big_ids:
				continue
			course_big_name[bi_id] = normalize_whitespace(bi.get("name", "") or "")
			course_big_desc[bi_id] = normalize_whitespace(bi.get("description", "") or "")
	
	# Skill descriptions
	skill_desc_lines: List[str] = []
	if include_skill_descriptions and used_skill_codes:
		for code in sorted(used_skill_codes):
			desc = trunc(skill_lookup.get(code, {}).get("description", ""), max_skill_desc_chars)
			skill_desc_lines.append(f"{code}={desc}" if desc else code)
	
	# Topic big idea descriptions
	topic_big_desc_lines: List[str] = []
	if include_topic_big_idea_descriptions and used_topic_big_ids:
		for bi_id in sorted(used_topic_big_ids):
			desc = topic_big_desc.get(bi_id, "")
			if not desc:
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
	
	# Build final unit context
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
			if name:
				unit_context_parts.append(f"{bi_id} ({name}): {desc}")
			else:
				unit_context_parts.append(f"{bi_id}: {desc}")
	
	# Topic big ideas
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


# Gemini call to process a single set
async def process_single_set(
	sem: asyncio.Semaphore,
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
	"""Process a single FRQ set with async generation and repair loop."""
	context_label = f"{course_id} | U{unit_index+1} | Set{set_index+1}"
	unit_title = unit.get("name", "")
	
	# Wait for permission from Semaphore (Rate Limit Guard)
	async with sem:
		log(f"[{context_label}] Starting FRQ generation...")
		all_frqs = []
		
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
			num_frqs=FRQS_PER_SET,
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
		all_frqs.extend(valid)
		
		# Track all invalid reports for error summary
		all_invalid_reports = invalid_initial.copy()
		
		# ----------------------------------------
		# 2. Repair Loop
		# ----------------------------------------
		repair_round = 0
		while len(all_frqs) < FRQS_PER_SET and repair_round < MAX_RETRIES_PER_SET:
			repair_round += 1
			missing = FRQS_PER_SET - len(all_frqs)
			
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
				num_frqs=request_count,
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
				all_frqs.extend(valid_repair)
				all_invalid_reports.extend(invalid_repair)  # Accumulate for next repair
			
			except Exception as e:
				log(f"[{context_label}] Repair API Error: {e}")
				break

	# ----------------------------------------
	# 3. Final Check & Save
	# ----------------------------------------
	if len(all_frqs) >= FRQS_PER_SET:
		all_frqs = all_frqs[:FRQS_PER_SET]
		
		# ----------------------------------------
		# 4. Generate Images for Stimuli
		# ----------------------------------------
		log(f"[{context_label}] Starting image generation for FRQs with stimulus_type='image'...")
		images_generated = 0
		images_failed = 0
		
		for frq_idx, frq in enumerate(all_frqs):
			if frq.get("stimulus_type") == "image":
				content = frq.get("stimulus_content", "")
				
				if content.startswith("IMAGE_PROMPT:"):
					raw_prompt = content.replace("IMAGE_PROMPT:", "").strip()
					
					# Enhance prompt for better image generation with full context
					question_stem = frq.get("context", "")  # FRQ uses "context" field
					enhanced_prompt = enhance_prompt_for_image_generation(
						raw_prompt=raw_prompt,
						question_stem=question_stem,
						question_type="FRQ",
						course_name=course_name,
						answer_choices=""  # FRQs don't have answer choices
					)
					
					# Create filename
					filename = create_image_filename(
						course_name=course_name,
						question_type="frq",
						unit_id=unit["id"],
						set_index=set_index,
						question_index=frq_idx
					)
					
					# Generate and save image
					image_path = OUTPUT_DIR / course_name / f"unit_{unit['id']}" / "images" / filename
					success, base64_data = await generate_image_from_prompt(
						prompt=enhanced_prompt,
						output_path=image_path,
						api_key=os.getenv("GOOGLE_API_KEY"),
						aspect_ratio="1:1"  # Can adjust based on stimulus type if needed
					)
					
					if success:
						# Update FRQ with image data
						frq["stimulus"] = {
							"type": "image",
							"file_path": str(image_path),
							"base64": base64_data,
							"alt_text": raw_prompt  # Use original prompt as alt text
						}
						images_generated += 1
						log(f"[{context_label}] Generated image: {filename}")
					else:
						images_failed += 1
						log(f"[{context_label}] WARNING: Failed to generate image for FRQ{frq_idx+1}")
						# Keep FRQ but mark stimulus as failed
						frq["stimulus"] = {
							"type": "none",
							"error": "Image generation failed"
						}
		
		log(f"[{context_label}] Image generation complete: {images_generated} success, {images_failed} failed")
		
		assign_frq_ids(
			all_frqs,
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
			frqs=all_frqs,
			context_label=context_label
		)
		
		# Update coverage tracker
		for frq in all_frqs:
			for lo_id in frq.get("aligned_lo_ids", []):
				if lo_id in coverage_tracker:
					coverage_tracker[lo_id] += 1
		
		log(f"[{context_label}] ✅ SUCCESS: Saved {len(all_frqs)} FRQs")
	else:
		log(f"[{context_label}] ❌ FAILED: Only got {len(all_frqs)}/{FRQS_PER_SET}")


# TSV parsing
def parse_tsv(tsv_text: str, context_label: str) -> List[List[str]]:
	"""Parse TSV text into rows of columns."""
	lines = [ln for ln in (tsv_text or "").splitlines() if ln.strip()]
	log(f"[{context_label}] parse_tsv: {len(lines)} non-empty lines found")
	
	rows = []
	for i, ln in enumerate(lines, start=1):
		cols = ln.split("\t")
		rows.append(cols)
	
	return rows


# Render + save to HTML Template
def render_html(
	html_template,
	course_name,
	course_id,
	unit_title,
	unit_index,
	set_index,
	frqs,
	context_label: str
):
	"""Render FRQs to HTML and save to file."""
	out_dir = OUTPUT_DIR / course_id / "frq"
	ensure_dir(out_dir)
	
	html = html_template.render(
		course=course_name,
		unit=f"Unit {unit_index + 1}: {unit_title}",
		set_number=set_index + 1,
		frqs=frqs
	)
	
	path = out_dir / f"unit{unit_index + 1}-set{set_index + 1}.html"
	
	if path.exists():
		log(f"[{context_label}] Skipping existing file: {path}")
		return path
	
	path.write_text(html, encoding="utf-8")
	log(f"[{context_label}] Wrote file: {path}")
	return path


def assign_frq_ids(
	frqs: List[dict],
	course_id: str,
	unit_index: int,
	set_index: int
) -> None:
	"""Assign unique IDs to FRQs in format: {course_id}_FRQ_U{unit}S{set}Q{num}"""
	for i, frq in enumerate(frqs, start=1):
		frq["id"] = f"{course_id}_FRQ_U{unit_index + 1}S{set_index + 1}Q{i}"


# Main Async Process
async def main_async():
	"""Main entry point for async FRQ generation."""
	client = init_client()
	
	# Load templates ONCE
	prompt_template = Template(load_text(PROMPT_PATH))
	repair_prompt_template = Template(load_text(REPAIR_PROMPT_PATH))
	html_template = Template(load_text(HTML_TEMPLATE_PATH))
	
	# Semaphore: adjust as needed
	sem = asyncio.Semaphore(60)
	
	tasks = []
	
	for course_name, course_id in AP_COURSES.items():
		course_spec = load_json(CONTENT_DIR / f"{course_id}.json")
		skill_lookup = build_skill_lookup(course_spec)
		big_idea_lookup = build_big_idea_lookup(course_spec)
		
		for unit_index, unit in enumerate(course_spec.get("units", [])):

			if unit_index != 6:
				continue

			# Initialize coverage tracker per unit
			coverage_tracker = initialize_lo_coverage(unit)

			# Build unit context ONCE per unit
			unit_context, constraints = build_unit_context(
				course_spec,
				unit,
				unit_index,
				skill_lookup,
				big_idea_lookup,
				question_type="frq"
			)
			
			for set_index in range(NUM_SETS_PER_UNIT):
				tasks.append(
					process_single_set(
						sem,
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
	
	print(f"Starting {len(tasks)} parallel FRQ tasks...")
	await asyncio.gather(*tasks)


if __name__ == "__main__":
	asyncio.run(main_async())

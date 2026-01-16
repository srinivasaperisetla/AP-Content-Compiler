import os
import json
import re
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from dotenv import load_dotenv
from google import genai

#Logging Utilities
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


# Gemini client initialization
def init_client() -> genai.Client:
	load_dotenv()
	api_key = os.getenv("GEMINI_API_KEY")
	assert api_key, "Missing GEMINI_API_KEY"
	google_client = genai.Client(api_key=api_key)
	print("Gemini client initialized")
	return google_client


# File and Test Helpers
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


# Course metadata helpers
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


# SVG validation
# Error summarization for repair prompts
def summarize_invalid_reports(invalid_reports: List[dict]) -> str:
	"""
	Generate detailed, actionable error summary for repair prompt.
	
	Args:
		invalid_reports: List of {row_index, reason, detail} dicts
	
	Returns:
		Human-readable, actionable error summary string
	"""
	if not invalid_reports:
		return "No specific errors recorded."
	
	# Group by reason and collect unique details
	reason_groups = {}
	for report in invalid_reports:
		reason = report.get("reason", "unknown")
		detail = report.get("detail", "")
		
		if reason not in reason_groups:
			reason_groups[reason] = []
		if detail and detail not in reason_groups[reason]:
			reason_groups[reason].append(detail)
	
	# Build actionable summary with specific guidance
	lines = ["Common errors from previous attempts:"]
	
	for reason, details in sorted(reason_groups.items(), key=lambda x: -len(x[1])):
		count = len(details)
		
		# Add specific, actionable guidance per error type
		if reason == "row_invalid":
			lines.append(f"- {count} rows had structural/format errors")
			lines.append("  → Each row MUST have exactly 11 tab-separated columns (MCQ) or 8 columns (FRQ)")
			lines.append("  → Use TAB character (\\t) between columns, NOT spaces")
			lines.append("  → Ensure difficulty is easy/medium/hard")
			lines.append("  → Ensure correct_idx is 0/1/2/3 (MCQ)")
		
		elif reason == "skill_not_allowed":
			invalid_skills = list(set(details))[:8]  # Show up to 8 unique invalid skills
			lines.append(f"- {count} questions used INVALID skill codes")
			lines.append(f"  → Invalid codes found: {', '.join(invalid_skills)}")
			lines.append("  → ONLY use codes from ALLOWED_SKILLS in unit context")
			lines.append("  → Double-check each skill code before using")
		
		elif reason == "lo_not_allowed":
			invalid_los = list(set(details))[:8]  # Show up to 8 unique invalid LOs
			lines.append(f"- {count} questions used INVALID learning objective IDs")
			lines.append(f"  → Invalid IDs found: {', '.join(invalid_los)}")
			lines.append("  → ONLY use IDs from ALLOWED_LOS in unit context")
			lines.append("  → Cross-reference each LO ID carefully")
		
		elif reason == "svg_invalid":
			lines.append(f"- {count} questions had invalid SVG graphics")
			lines.append("  → SVG MUST start with <svg and end with </svg>")
			lines.append("  → NO <script> tags allowed (security)")
			lines.append("  → Font size must be ≥ 12")
			lines.append("  → Maximum 8 <text> elements")
		
		elif reason == "table_invalid":
			lines.append(f"- {count} questions had invalid table format")
			lines.append("  → ALL rows must start AND end with | character")
			lines.append("  → Second row MUST be separator: | --- | --- |")
			lines.append("  → ALL rows must have SAME column count")
			lines.append("  → Use literal \\n between rows (not actual newlines)")
		
		else:
			# Generic fallback
			if details:
				lines.append(f"- {reason} ({count}x): {details[0][:60]}")
			else:
				lines.append(f"- {reason} ({count}x)")
	
	lines.append("")
	lines.append("CRITICAL REMINDER: Generate COMPLETE, VALID TSV rows following ALL format rules.")
	
	return "\n".join(lines)


# LO description compression
def compress_lo_description(text: str) -> str:
	"""
	Compress LO descriptions using rule-based abbreviation.
	Preserves meaning while reducing token count.
	
	Example:
		"Identify individuals, variables, and categorical or quantitative variables"
		→ "Identify individuals/variables (categorical vs quantitative)"
	"""
	if not text:
		return text
	
	# Apply compression rules (order matters)
	compressed = text
	
	# Replace verbose conjunctions
	compressed = re.sub(r'\s+and/or\s+', '/', compressed)
	compressed = re.sub(r'\s+or\s+', ' vs ', compressed)
	compressed = re.sub(r',\s+and\s+', '/', compressed)
	
	# Common statistical phrases
	compressed = compressed.replace("categorical or quantitative", "categorical vs quantitative")
	compressed = compressed.replace("provided conditions for inference are met", "if conditions met")
	compressed = compressed.replace("using evidence and/or reasoning", "using evidence/reasoning")
	compressed = compressed.replace("describe the distribution of", "describe distribution of")
	compressed = compressed.replace("identify an appropriate", "identify appropriate")
	compressed = compressed.replace("in a given situation", "in situation")
	compressed = compressed.replace("in the context of", "in context of")
	
	# Remove filler phrases at end (preserve if mid-sentence)
	if compressed.endswith(" in context"):
		pass  # Keep if it's the ending
	else:
		compressed = compressed.replace(" in context,", ",")
		compressed = compressed.replace(" in context ", " ")
	
	# Simplify "A, B, and C" patterns
	compressed = re.sub(r'(\w+),\s+(\w+),\s+and\s+(\w+)', r'\1/\2/\3', compressed)
	
	# Clean up spacing
	compressed = re.sub(r'\s+', ' ', compressed).strip()
	
	return compressed


# Coverage tracking for LO distribution
def initialize_lo_coverage(unit: dict) -> Dict[str, int]:
	"""
	Initialize LO coverage tracker for a unit.
	Returns dict mapping LO_ID -> usage_count (all start at 0).
	"""
	coverage = {}
	for topic in unit.get("topics", []):
		for lo in topic.get("learning_objectives", []):
			lo_id = str(lo.get("id", "")).strip()
			if lo_id:
				coverage[lo_id] = 0
	return coverage


def get_priority_los(coverage: Dict[str, int], allowed_los: List[str], top_n: int = None) -> List[str]:
	"""
	Get under-covered LOs to prioritize in next generation.
	
	Args:
		coverage: Current LO usage counts
		allowed_los: LOs allowed for this unit
		top_n: Return top N least-covered (default: bottom 50%)
	
	Returns:
		List of LO IDs sorted by coverage (ascending)
	"""
	if not coverage:
		return allowed_los
	
	# Get coverage counts for allowed LOs
	lo_counts = [(lo, coverage.get(lo, 0)) for lo in allowed_los]
	
	# Sort by count (ascending)
	lo_counts.sort(key=lambda x: x[1])
	
	# Return bottom N
	if top_n is None:
		top_n = max(len(lo_counts) // 2, 1)  # Bottom 50%
	
	return [lo for lo, _ in lo_counts[:top_n]]


# Global constants (used by log_block and log_context)
DEBUG = True
NUM_SETS_PER_UNIT = 20
#!/usr/bin/env python3
"""
Phase 2: Retrieve batch job results and render HTML files.

This script:
1. Loads all pending batch jobs from state files
2. Polls each job for completion status
3. Downloads images for completed jobs
4. Renders HTML files with images embedded
5. Cleans up state files for completed jobs

Usage:
    python batch_retrieve_and_render.py
"""

import os
import time
import base64
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from google import genai
from jinja2 import Template
import certifi

from utility_functions import (
    log,
    init_client,
    load_text,
    ensure_dir
)

from utils.batch_state_manager import (
    load_pending_batch_jobs,
    mark_batch_job_completed,
    get_state_summary
)

from utils.batch_image_generator import (
    check_batch_job_status,
    retrieve_batch_results,
    get_batch_job_info
)


# Load environment
load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()

# Paths
OUTPUT_DIR = Path("output")
TEMPLATE_DIR = Path("utils/templates")


def assign_question_ids_for_set(
    questions: List[dict],
    course_id: str,
    unit_index: int,
    set_index: int,
    question_type: str
) -> None:
    """Assign IDs to questions."""
    for i, q in enumerate(questions, start=1):
        if question_type == "mcq":
            q["id"] = f"{course_id}_MCQ_U{unit_index + 1}S{set_index + 1}Q{i}"
        else:  # frq
            q["id"] = f"{course_id}_FRQ_U{unit_index + 1}S{set_index + 1}Q{i}"


def render_html_mcq(
    course_name: str,
    course_id: str,
    unit_title: str,
    unit_index: int,
    set_index: int,
    questions: List[dict],
    html_template: Template
) -> Path:
    """Render MCQ HTML file."""
    out_dir = OUTPUT_DIR / course_id / "mcq"
    ensure_dir(out_dir)
    
    html = html_template.render(
        course=course_name,
        unit=f"Unit {unit_index + 1}: {unit_title}",
        questions=questions
    )
    
    path = out_dir / f"unit{unit_index + 1}-set{set_index + 1}.html"
    path.write_text(html, encoding="utf-8")
    
    return path


def render_html_frq(
    course_name: str,
    course_id: str,
    unit_title: str,
    unit_index: int,
    set_index: int,
    frqs: List[dict],
    html_template: Template
) -> Path:
    """Render FRQ HTML file."""
    out_dir = OUTPUT_DIR / course_id / "frq"
    ensure_dir(out_dir)
    
    html = html_template.render(
        course=course_name,
        unit=f"Unit {unit_index + 1}: {unit_title}",
        set_number=set_index + 1,
        frqs=frqs
    )
    
    path = out_dir / f"unit{unit_index + 1}-set{set_index + 1}.html"
    path.write_text(html, encoding="utf-8")
    
    return path


def process_completed_job(client: genai.Client, job_state, html_templates: Dict[str, Template]):
    """Process a completed batch job - download images and render HTML."""
    log(f"\n[{job_state.course_name} Unit {job_state.unit_index + 1}] Processing completed batch job...")
    log(f"  Job: {job_state.job_name}")
    log(f"  Total images: {job_state.total_image_requests}")
    
    # Download all images for this unit
    output_dir = OUTPUT_DIR / "images" / job_state.course_name / f"unit_{job_state.unit_index+1}"
    
    try:
        image_results = retrieve_batch_results(client, job_state.job_name, output_dir)
        log(f"  ✓ Downloaded {len(image_results)} images")
    except Exception as e:
        log(f"  ✗ Error retrieving images: {e}")
        return 0
    
    # Process each set in this unit
    html_files_rendered = 0
    html_template = html_templates[job_state.question_type]
    
    for set_data_dict in job_state.sets:
        set_index = set_data_dict["set_index"]
        questions_data = set_data_dict["questions_data"]
        image_requests = set_data_dict["image_requests"]
        
        # Update questions with image data
        for req in image_requests:
            q_idx = req["question_index"]
            key = req["key"]  # Format: u1_s3_q5
            
            if key in image_results:
                image_path = image_results[key]
                image_bytes = image_path.read_bytes()
                base64_data = base64.b64encode(image_bytes).decode('utf-8')
                
                questions_data[q_idx]["stimulus"] = {
                    "type": "image",
                    "file_path": str(image_path),
                    "base64": base64_data,
                    "alt_text": req.get("prompt", "Generated image")
                }
            else:
                # Image failed - mark as error
                log(f"  ⚠ Missing image for {key}")
                questions_data[q_idx]["stimulus"] = {
                    "type": "none",
                    "error": "Image generation failed"
                }
        
        # Assign IDs
        assign_question_ids_for_set(
            questions_data,
            job_state.course_id,
            job_state.unit_index,
            set_index,
            job_state.question_type
        )
        
        # Render HTML
        try:
            if job_state.question_type == "mcq":
                html_path = render_html_mcq(
                    job_state.course_name,
                    job_state.course_id,
                    job_state.unit_title,
                    job_state.unit_index,
                    set_index,
                    questions_data,
                    html_template
                )
            else:  # frq
                html_path = render_html_frq(
                    job_state.course_name,
                    job_state.course_id,
                    job_state.unit_title,
                    job_state.unit_index,
                    set_index,
                    questions_data,
                    html_template
                )
            
            log(f"  ✓ Rendered: {html_path}")
            html_files_rendered += 1
            
        except Exception as e:
            log(f"  ✗ Error rendering HTML for set {set_index + 1}: {e}")
    
    log(f"  ✓ Total HTML files rendered: {html_files_rendered}")
    return html_files_rendered


def main():
    """Main entry point for Phase 2 processing."""
    log("=" * 60)
    log("PHASE 2: Batch Job Retrieval & HTML Rendering")
    log("=" * 60)
    
    # Initialize client
    client = init_client()
    
    # Load HTML templates
    mcq_template_path = TEMPLATE_DIR / "mcq.html"
    frq_template_path = TEMPLATE_DIR / "frq.html"
    
    html_templates = {
        "mcq": Template(load_text(mcq_template_path)),
        "frq": Template(load_text(frq_template_path))
    }
    
    # Load pending batch jobs
    batch_dir = Path("batch_jobs/state")
    pending_jobs = load_pending_batch_jobs(batch_dir)
    
    if len(pending_jobs) == 0:
        log("\n✓ No pending batch jobs found!")
        log("All jobs have been processed.")
        return
    
    # Print summary
    summary = get_state_summary(batch_dir)
    log(f"\nFound {summary['total_jobs']} pending batch job(s):")
    log(f"  Total images queued: {summary['total_images']}")
    log(f"  MCQ jobs: {summary['by_type']['mcq']}")
    log(f"  FRQ jobs: {summary['by_type']['frq']}")
    log(f"\nBy course:")
    for course_id, course_info in summary['by_course'].items():
        log(f"  {course_id}: {course_info['jobs']} job(s), {course_info['images']} images")
    
    # Polling configuration
    completed_states = {
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED'
    }
    
    poll_interval = 10  # seconds
    total_rendered = 0
    total_failed = 0
    
    log("\n" + "=" * 60)
    log("Starting polling...")
    log("=" * 60)
    
    # Main polling loop
    while pending_jobs:
        for job_state in list(pending_jobs):
            try:
                status = check_batch_job_status(client, job_state.job_name)
                
                if status not in completed_states:
                    log(f"[{job_state.course_name} U{job_state.unit_index+1}] {status}")
                    continue
                
                # Job completed - process it
                if status == 'JOB_STATE_SUCCEEDED':
                    html_count = process_completed_job(client, job_state, html_templates)
                    total_rendered += html_count
                else:
                    log(f"\n[{job_state.course_name} U{job_state.unit_index+1}] ✗ Job failed: {status}")
                    total_failed += 1
                
                # Mark as completed and remove from list
                mark_batch_job_completed(job_state.job_name, batch_dir)
                pending_jobs.remove(job_state)
                
            except Exception as e:
                log(f"\n[Error] Failed to process job {job_state.job_name}: {e}")
                # Keep in list to retry
        
        # Wait before next poll if jobs remaining
        if pending_jobs:
            remaining = len(pending_jobs)
            log(f"\nWaiting {poll_interval}s before next poll... ({remaining} job(s) remaining)")
            time.sleep(poll_interval)
    
    # Final summary
    log("\n" + "=" * 60)
    log("PHASE 2 COMPLETE")
    log("=" * 60)
    log(f"HTML files rendered: {total_rendered}")
    log(f"Failed jobs: {total_failed}")
    log("\n✓ All batch jobs processed!")


if __name__ == "__main__":
    main()

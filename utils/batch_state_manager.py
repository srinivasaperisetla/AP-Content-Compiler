"""
Batch State Manager - Manages persistent state for batch jobs between phases.

This module handles saving and loading batch job state to/from disk, allowing
the pipeline to be split into two phases: text generation + batch submission
(Phase 1) and image retrieval + HTML rendering (Phase 2).
"""

from pathlib import Path
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime


@dataclass
class SetData:
    """Data for a single set within a unit."""
    set_index: int
    questions_data: List[Dict[str, Any]]  # All questions for this set
    image_requests: List[Dict[str, Any]]  # Image requests for this set (key, prompt, question_index)


@dataclass
class BatchJobState:
    """State for ONE batch job covering ALL sets in a unit."""
    job_name: str
    course_id: str
    course_name: str
    unit_index: int
    unit_title: str
    question_type: str  # "mcq" or "frq"
    sets: List[Dict[str, Any]]  # Serialized SetData objects
    total_image_requests: int  # Total images across all sets
    jsonl_file_path: str
    uploaded_file_name: str
    created_at: str


def save_batch_job_state(state: BatchJobState, batch_dir: Path) -> Path:
    """
    Save batch job state to JSON file.
    
    Args:
        state: BatchJobState object to save
        batch_dir: Directory to save state files
        
    Returns:
        Path to saved state file
    """
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename from course and unit
    filename = f"{state.course_id}_u{state.unit_index + 1}_{state.question_type}.json"
    state_file = batch_dir / filename
    
    # Convert to dict for JSON serialization
    state_dict = asdict(state)
    
    # Write to file
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state_dict, f, indent=2, ensure_ascii=False)
    
    return state_file


def load_pending_batch_jobs(batch_dir: Path) -> List[BatchJobState]:
    """
    Load all pending batch jobs from state files.
    
    Args:
        batch_dir: Directory containing state files
        
    Returns:
        List of BatchJobState objects
    """
    if not batch_dir.exists():
        return []
    
    pending_jobs = []
    
    for state_file in batch_dir.glob("*.json"):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state_dict = json.load(f)
            
            # Reconstruct BatchJobState from dict
            state = BatchJobState(**state_dict)
            pending_jobs.append(state)
            
        except Exception as e:
            print(f"Error loading state file {state_file}: {e}")
            continue
    
    return pending_jobs


def mark_batch_job_completed(job_name: str, batch_dir: Path) -> bool:
    """
    Mark batch job as completed by deleting its state file.
    
    Args:
        job_name: Name of the batch job
        batch_dir: Directory containing state files
        
    Returns:
        True if file was deleted, False if not found
    """
    if not batch_dir.exists():
        return False
    
    # Find state file by job_name
    for state_file in batch_dir.glob("*.json"):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state_dict = json.load(f)
            
            if state_dict.get('job_name') == job_name:
                state_file.unlink()
                return True
                
        except Exception as e:
            print(f"Error processing state file {state_file}: {e}")
            continue
    
    return False


def get_state_summary(batch_dir: Path) -> Dict[str, Any]:
    """
    Get summary of pending batch jobs.
    
    Args:
        batch_dir: Directory containing state files
        
    Returns:
        Dict with summary statistics
    """
    pending_jobs = load_pending_batch_jobs(batch_dir)
    
    summary = {
        "total_jobs": len(pending_jobs),
        "total_images": sum(job.total_image_requests for job in pending_jobs),
        "by_course": {},
        "by_type": {"mcq": 0, "frq": 0}
    }
    
    for job in pending_jobs:
        # Count by course
        if job.course_id not in summary["by_course"]:
            summary["by_course"][job.course_id] = {
                "jobs": 0,
                "images": 0,
                "units": []
            }
        summary["by_course"][job.course_id]["jobs"] += 1
        summary["by_course"][job.course_id]["images"] += job.total_image_requests
        summary["by_course"][job.course_id]["units"].append(job.unit_index + 1)
        
        # Count by type
        summary["by_type"][job.question_type] += 1
    
    return summary

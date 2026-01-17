"""
Batch Image Generator - Handles batch API interactions for image generation.

This module provides functions to create batch jobs, check their status, and
retrieve results from Google's Batch API for image generation.
"""

import json
import base64
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from google import genai
from google.genai import types


# Model for batch image generation
IMAGE_MODEL = "gemini-3-pro-image-preview"


async def create_batch_job_for_unit(
    client: genai.Client,
    image_requests: List[Dict],
    unit_label: str,
    model: str = IMAGE_MODEL
) -> Tuple[str, str, Path]:
    """
    Create a batch job for a unit's image generation requests.
    
    Args:
        client: Gemini client
        image_requests: List of dicts with keys: 'key', 'prompt', 'question_index'
        unit_label: Display name for the job (e.g., "ap_physics_1_u1")
        model: Model to use for image generation
        
    Returns:
        Tuple of (batch_job_name, uploaded_file_name, jsonl_file_path)
    """
    # 1. Create JSONL file
    jsonl_path = create_jsonl_file(image_requests, unit_label)
    
    # 2. Upload file to Google
    loop = asyncio.get_event_loop()
    uploaded_file = await loop.run_in_executor(
        None,
        lambda: client.files.upload(
            file=str(jsonl_path),
            config=types.UploadFileConfig(
                display_name=f'{unit_label}-batch-requests',
                mime_type='application/jsonl'
            )
        )
    )
    
    # 3. Create batch job
    batch_job = await loop.run_in_executor(
        None,
        lambda: client.batches.create(
            model=f"models/{model}",
            src=uploaded_file.name,
            config={'display_name': unit_label}
        )
    )
    
    return batch_job.name, uploaded_file.name, jsonl_path


def create_jsonl_file(image_requests: List[Dict], unit_label: str) -> Path:
    """
    Create JSONL file with image generation requests.
    
    Args:
        image_requests: List of dicts with 'key' and 'prompt'
        unit_label: Label for the unit (used in filename)
        
    Returns:
        Path to created JSONL file
    """
    jsonl_dir = Path("batch_jobs") / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    
    jsonl_path = jsonl_dir / f"{unit_label}.jsonl"
    
    with open(jsonl_path, "w", encoding='utf-8') as f:
        for req in image_requests:
            jsonl_request = {
                "key": req["key"],
                "request": {
                    "contents": [{"parts": [{"text": req["prompt"]}]}],
                    "generation_config": {
                        "responseModalities": ["IMAGE"]
                    }
                }
            }
            f.write(json.dumps(jsonl_request, ensure_ascii=False) + "\n")
    
    return jsonl_path


def check_batch_job_status(client: genai.Client, job_name: str) -> str:
    """
    Check status of batch job synchronously.
    
    Args:
        client: Gemini client
        job_name: Name of the batch job
        
    Returns:
        State name (e.g., 'JOB_STATE_SUCCEEDED', 'JOB_STATE_PENDING')
    """
    batch_job = client.batches.get(name=job_name)
    return batch_job.state.name


def retrieve_batch_results(
    client: genai.Client,
    job_name: str,
    output_dir: Path
) -> Dict[str, Path]:
    """
    Retrieve results from completed batch job and save images.
    
    Args:
        client: Gemini client
        job_name: Name of the batch job
        output_dir: Directory to save images
        
    Returns:
        Dict mapping request keys to saved image paths
        
    Raises:
        Exception: If job is not in succeeded state
    """
    batch_job = client.batches.get(name=job_name)
    
    if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
        raise Exception(f"Job not succeeded: {batch_job.state.name}")
    
    # Get result file
    result_file_name = batch_job.dest.file_name
    file_content_bytes = client.files.download(file=result_file_name)
    file_content = file_content_bytes.decode('utf-8')
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse results and save images
    image_results = {}
    
    for line in file_content.splitlines():
        if not line.strip():
            continue
        
        try:
            parsed = json.loads(line)
            request_key = parsed.get('key')
            
            if not request_key:
                continue
            
            # Check for response
            if 'response' in parsed and parsed['response']:
                # Extract image from response
                try:
                    candidates = parsed['response'].get('candidates', [])
                    if candidates:
                        parts = candidates[0].get('content', {}).get('parts', [])
                        
                        for part in parts:
                            if part.get('inlineData'):
                                # Decode and save image
                                image_data = base64.b64decode(part['inlineData']['data'])
                                image_path = output_dir / f"{request_key}.jpeg"
                                image_path.write_bytes(image_data)
                                image_results[request_key] = image_path
                                break
                                
                except Exception as e:
                    print(f"Error processing image for key {request_key}: {e}")
                    
            elif 'error' in parsed:
                print(f"Error for key {request_key}: {parsed['error']}")
                
        except json.JSONDecodeError as e:
            print(f"Error parsing line: {e}")
            continue
    
    return image_results


def get_batch_job_info(client: genai.Client, job_name: str) -> Dict:
    """
    Get detailed information about a batch job.
    
    Args:
        client: Gemini client
        job_name: Name of the batch job
        
    Returns:
        Dict with job information
    """
    batch_job = client.batches.get(name=job_name)
    
    info = {
        "name": batch_job.name,
        "state": batch_job.state.name,
        "display_name": getattr(batch_job, 'display_name', 'N/A'),
        "create_time": str(getattr(batch_job, 'create_time', 'N/A')),
        "start_time": str(getattr(batch_job, 'start_time', 'N/A')),
        "end_time": str(getattr(batch_job, 'end_time', 'N/A')),
    }
    
    # Add batch stats if available
    if hasattr(batch_job, 'batch_stats'):
        stats = batch_job.batch_stats
        info["stats"] = {
            "total": getattr(stats, 'total_request_count', 0),
            "succeeded": getattr(stats, 'succeeded_request_count', 0),
            "failed": getattr(stats, 'failed_request_count', 0),
        }
    
    return info

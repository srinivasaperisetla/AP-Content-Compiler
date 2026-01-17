#!/usr/bin/env python3
"""
Cleanup script for HTML files with failed image generation.

This script scans the output directory for HTML files containing image generation
errors and moves them to a junk folder for manual review.
"""

from pathlib import Path
import shutil
from bs4 import BeautifulSoup


def has_failed_images(html_file: Path) -> bool:
    """
    Check if an HTML file contains failed image generation errors.
    
    Args:
        html_file: Path to HTML file to check
        
    Returns:
        True if file contains stimulus-error divs indicating failed images
    """
    try:
        html_content = html_file.read_text(encoding='utf-8')
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Look for divs with class 'stimulus-error' containing failure text
        error_divs = soup.find_all('div', class_='stimulus-error')
        
        for div in error_divs:
            text = div.get_text()
            if 'Image generation failed' in text or 'Visual stimulus unavailable' in text:
                return True
        
        return False
    except Exception as e:
        print(f"Error reading {html_file}: {e}")
        return False


def scan_and_move_failed_html_files():
    """
    Scan all HTML files in output directory and move ones with failed images to junk folder.
    """
    output_dir = Path("output")
    junk_dir = output_dir / "junk"
    
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return
    
    total_moved = 0
    moved_by_course = {}
    
    # Scan each course directory
    for course_dir in output_dir.iterdir():
        if not course_dir.is_dir():
            continue
            
        # Skip special directories
        if course_dir.name in ["images", "junk"]:
            continue
        
        course_name = course_dir.name
        moved_by_course[course_name] = {"mcq": 0, "frq": 0}
        
        # Check both MCQ and FRQ directories
        for question_type in ["mcq", "frq"]:
            type_dir = course_dir / question_type
            if not type_dir.exists():
                continue
            
            # Scan all HTML files in this directory
            for html_file in type_dir.glob("*.html"):
                if has_failed_images(html_file):
                    # Create destination directory
                    dest_dir = junk_dir / course_name / question_type
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Move file to junk folder
                    dest_path = dest_dir / html_file.name
                    
                    try:
                        shutil.move(str(html_file), str(dest_path))
                        print(f"Moved: {html_file.relative_to(output_dir)} -> {dest_path.relative_to(output_dir)}")
                        total_moved += 1
                        moved_by_course[course_name][question_type] += 1
                    except Exception as e:
                        print(f"Error moving {html_file}: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("CLEANUP SUMMARY")
    print("=" * 60)
    
    if total_moved == 0:
        print("✓ No HTML files with failed images found!")
    else:
        print(f"Total files moved to junk: {total_moved}\n")
        print("Breakdown by course:")
        for course_name, counts in moved_by_course.items():
            mcq_count = counts["mcq"]
            frq_count = counts["frq"]
            if mcq_count > 0 or frq_count > 0:
                print(f"  {course_name}:")
                if mcq_count > 0:
                    print(f"    MCQ: {mcq_count} file(s)")
                if frq_count > 0:
                    print(f"    FRQ: {frq_count} file(s)")
        
        print(f"\nMoved files are now in: {junk_dir}")


def main():
    """Main entry point for the cleanup script."""
    print("Starting cleanup of HTML files with failed image generation...")
    print("=" * 60)
    scan_and_move_failed_html_files()


if __name__ == "__main__":
    main()

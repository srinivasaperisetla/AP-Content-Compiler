#!/usr/bin/env python3
"""
Rename MCQ HTML files from nested structure to flat format:
From: output/ap_statistics/unit_X/mcqs/set_Y.html
To: output/ap_statistics/unitX-setY.html
"""

import os
import re
from pathlib import Path

OUTPUT_DIR = "output/ap_statistics"

def rename_mcq_files():
	"""Rename all MCQ HTML files to the new format."""
	
	if not os.path.exists(OUTPUT_DIR):
		print(f"❌ Output directory not found: {OUTPUT_DIR}")
		return
	
	# Find all HTML files in the nested structure
	html_files = []
	for root, dirs, files in os.walk(OUTPUT_DIR):
		for file in files:
			if file.endswith(".html") and file.startswith("set_"):
				html_files.append(os.path.join(root, file))
	
	if not html_files:
		print(f"❌ No HTML files found in {OUTPUT_DIR}")
		return
	
	print(f"📁 Found {len(html_files)} HTML files to rename")
	
	# Extract unit and set numbers, then rename
	renamed_count = 0
	errors = []
	
	for old_path in html_files:
		try:
			# Extract unit number from path: output/ap_statistics/unit_X/mcqs/set_Y.html
			path_parts = old_path.split(os.sep)
			
			# Find unit_X directory
			unit_match = None
			set_match = None
			
			# Extract unit and set numbers from the full path
			unit_match = re.search(r"unit_(\d+)", old_path)
			set_match = re.search(r"set_(\d+)", old_path)
			
			if not unit_match or not set_match:
				errors.append(f"Could not extract unit/set from: {old_path}")
				continue
			
			unit_num = unit_match.group(1)
			set_num = set_match.group(1)
			
			# Create new filename: unitX-setY.html
			new_filename = f"unit{unit_num}-set{set_num}.html"
			new_path = os.path.join(OUTPUT_DIR, new_filename)
			
			# Check if target already exists
			if os.path.exists(new_path):
				print(f"⚠️  Target already exists, skipping: {new_path}")
				continue
			
			# Rename (move) the file
			os.rename(old_path, new_path)
			renamed_count += 1
			print(f"✅ Renamed: {os.path.basename(old_path)} → {new_filename}")
			
		except Exception as e:
			errors.append(f"Error renaming {old_path}: {e}")
	
	print(f"\n📊 Summary:")
	print(f"   ✅ Successfully renamed: {renamed_count} files")
	if errors:
		print(f"   ❌ Errors: {len(errors)}")
		for error in errors[:10]:  # Show first 10 errors
			print(f"      {error}")
		if len(errors) > 10:
			print(f"      ... and {len(errors) - 10} more errors")
	
	# Clean up empty directories
	print(f"\n🧹 Cleaning up empty directories...")
	cleaned = 0
	for root, dirs, files in os.walk(OUTPUT_DIR, topdown=False):
		# Remove empty directories (except the base output/ap_statistics)
		if root != OUTPUT_DIR:
			try:
				if not os.listdir(root):
					os.rmdir(root)
					cleaned += 1
					print(f"   🗑️  Removed empty directory: {root}")
			except OSError:
				pass
	
	if cleaned > 0:
		print(f"   ✅ Cleaned up {cleaned} empty directories")

if __name__ == "__main__":
	rename_mcq_files()


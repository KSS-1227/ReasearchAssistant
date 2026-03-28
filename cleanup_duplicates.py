#!/usr/bin/env python3
"""
Script to remove duplicate method definitions in citation_extractor.py
This removes the second occurrence of duplicated methods
"""

import re

filepath = r'Agentic_AI_FINAL_Project-main\agents\citation_extractor.py'

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and remove the second _build_citation_network and _calculate_insights methods
# They appear around lines 871-943

# Strategy: Find the line numbers of the duplicate sections and remove them
output_lines = []
skip_section = False
i = 0

while i < len(lines):
    line = lines[i]
    
    # Check if this is the start of the duplicate _build_citation_network (second occurrence)
    # It should be preceded by year_range["max"] = max(year_range["max"], paper.year)
    # and followed by network = {
    if i > 850 and i < 900 and 'def _build_citation_network' in line:
        # Look back to confirm this is after _update_metadata_stats
        context_before = ''.join(lines[max(0, i-5):i])
        if 'year_range["max"]' in context_before and i > 850:
            # Found the duplicate section! Skip until we find the next def _calculate_recency_score
            print(f"Found duplicate _build_citation_network at line {i+1}")
            while i < len(lines) and 'def _calculate_recency_score' not in lines[i]:
                i += 1
            # Don't skip the _calculate_recency_score line itself
            continue
    
    output_lines.append(line)
    i += 1

# Write the cleaned file
with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(output_lines)

print(f"Cleanup complete! Removed duplicate methods from {filepath}")
print(f"Original line count: {len(lines)}")
print(f"New line count: {len(output_lines)}")
print(f"Lines removed: {len(lines) - len(output_lines)}")

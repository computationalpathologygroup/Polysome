#!/usr/bin/env python3
"""
Polysome Output Validation Script
Validates that integration test outputs meet quality and structure requirements.
"""

import json
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def validate_jsonl(file_path: Path, expected_count: int, keywords_map: dict = None) -> bool:
    """
    Validates a JSONL output file.
    """
    if not file_path.exists():
        logger.error(f"Missing output file: {file_path}")
        return False
        
    logger.info(f"Validating {file_path}...")
    
    records = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return False
        
    # Check count
    if len(records) != expected_count:
        logger.error(f"Expected {expected_count} records, found {len(records)} in {file_path.name}")
        return False
        
    # Check each record
    success = True
    for i, record in enumerate(records):
        # 1. Check required fields
        if 'id' not in record or 'question' not in record or 'answer' not in record:
            logger.error(f"Record {i} missing required fields (id, question, answer)")
            success = False
            continue
            
        # 2. Check answer quality
        answer = record.get('answer', '')
        if not answer or len(answer) < 10:
            logger.error(f"Record {record['id']} has empty or too short answer: '{answer}'")
            success = False
            
        # 3. Check keywords (optional warning)
        if keywords_map and record['id'] in keywords_map:
            expected = keywords_map[record['id']]
            if expected.lower() not in answer.lower():
                logger.warning(f"Record {record['id']} answer does not contain expected keyword '{expected}'")
                
    if success:
        logger.info(f"✅ {file_path.name} passed validation.")
    else:
        logger.error(f"❌ {file_path.name} failed validation.")
        
    return success

def main():
    project_root = Path(__file__).parent.parent.parent.parent
    test_root = Path(__file__).parent.parent
    criteria_path = test_root / "expected_outputs" / "validation_criteria.json"
    output_base = project_root / "output" / "integration_tests"
    
    if not criteria_path.exists():
        logger.error(f"Criteria file not found: {criteria_path}")
        sys.exit(1)
        
    with open(criteria_path, 'r') as f:
        criteria = json.load(f)
        
    # Validation criteria
    expected_count = criteria.get("expected_record_count", 5)
    keywords_map = criteria.get("keyword_expectations", {})
    
    engines_config = criteria.get("output_paths", {})
    
    all_success = True
    found_any = False
    
    for engine, rel_path in engines_config.items():
        path = output_base / rel_path
        if path.exists():
            found_any = True
            if not validate_jsonl(path, expected_count, keywords_map):
                all_success = False
        else:
            logger.info(f"Engine {engine} output not found at {path} (skipping)")
            
    if not found_any:
        logger.error("No integration test outputs found to validate!")
        sys.exit(1)
        
    if all_success:
        logger.info("=== All available outputs passed validation! ===")
        sys.exit(0)
    else:
        logger.error("=== Validation failed for one or more outputs. ===")
        sys.exit(1)

if __name__ == "__main__":
    main()

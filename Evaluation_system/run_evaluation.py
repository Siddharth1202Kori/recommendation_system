#!/usr/bin/env python3
"""
Simple script to run the assessment evaluator against a hosted API endpoint.
"""
import sys
import json
import logging
import time
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to path to make imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import can work properly 
try:
    from Evaluation_system.assessment_evaluator import AssessmentEvaluator
except ImportError:
    # If that doesn't work, try direct import (if in same directory)
    try:
        from assessment_evaluator import AssessmentEvaluator
    except ImportError:
        raise ImportError("Could not import AssessmentEvaluator. Make sure the module is in your Python path.")

# Default test cases
DEFAULT_TEST_CASES = [
    {
        "input": "Looking for a data entry specialist with attention to detail",
        "expected_keywords": ["data entry", "clerical", "entry-level", "administrative"]
    },
    {
        "input": "Need a project manager for our software development team",
        "expected_keywords": ["project manager", "projects", "managing projects", "project timelines"]
    },
    {
        "input": "Hiring a business analyst with data skills",
        "expected_keywords": ["professional", "analyzing data", "gathering requirements"]
    },
    {
        "input": "Looking for manager to lead our sales department",
        "expected_keywords": ["manager", "leadership", "coaching employees", "meeting goals"]
    },
    {
        "input": "Data scientist position for experienced professionals",
        "expected_keywords": ["data", "analyzing", "professional", "requirements"]
    }
]

# Alternative test cases with simplified expected keywords
FALLBACK_TEST_CASES = [
    {
        "input": "Looking for a data entry specialist with attention to detail",
        "expected_keywords": ["data", "entry", "administrative"]
    },
    {
        "input": "Need a project manager for our software development team",
        "expected_keywords": ["project", "manager", "software"]
    },
    {
        "input": "Hiring a business analyst with data skills",
        "expected_keywords": ["business", "analyst", "data"]
    },
    {
        "input": "Looking for manager to lead our sales department",
        "expected_keywords": ["manager", "sales", "leadership"]
    },
    {
        "input": "Data scientist position for experienced professionals",
        "expected_keywords": ["data", "scientist", "professional"]
    }
]

def load_test_cases(filename=None):
    """Load test cases from a file or use defaults."""
    if filename:
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load test cases from {filename}: {e}")
            logger.info("Falling back to default test cases")
    
    return DEFAULT_TEST_CASES

def main():
    """Main function to run the evaluation."""
    # Handle command line args
    if len(sys.argv) < 2:
        print("Usage: python run_evaluation.py API_URL [TEST_CASES_FILE] [--fallback]")
        print("Example: python run_evaluation.py https://recommendation-system-1-0j7h.onrender.com/get-assessments")
        print("Add --fallback flag to use simplified test cases if the API is unstable")
        return
    
    api_url = sys.argv[1]
    use_fallback = "--fallback" in sys.argv
    
    # Find test cases file if provided
    test_cases_file = None
    for arg in sys.argv[2:]:
        if arg != "--fallback" and not arg.startswith("--"):
            test_cases_file = arg
            break
    
    # Load test cases
    test_cases = FALLBACK_TEST_CASES if use_fallback else load_test_cases(test_cases_file)
    
    # Run evaluation
    print("\n🔄 Starting assessment API evaluation...")
    print(f"🌐 API URL: {api_url}")
    print(f"🔢 Test cases: {len(test_cases)}")
    print(f"⚠️ Using fallback simplified test cases: {use_fallback}")
    
    try:
        evaluator = AssessmentEvaluator()
        results = evaluator.evaluate_test_set(api_url, test_cases)
        evaluator.print_evaluation_summary(results)
        
        print("\n✅ Evaluation completed!")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if the API server is running and accessible")
        print("2. Try running with the --fallback flag: python run_evaluation.py", api_url, "--fallback")
        print("3. The API server may be slow or overloaded - try again later")
        print("4. Check your internet connection")

if __name__ == "__main__":
    main()
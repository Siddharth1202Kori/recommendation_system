import numpy as np
import json
import urllib.request
import urllib.error
import time
import logging
import argparse
from typing import List, Dict, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AssessmentEvaluator:
    """Evaluates assessment recommendations against expected keywords and semantic similarity."""
    
    def __init__(self):
        """Initialize the evaluator with simple text comparison functionality."""
        logger.info("Initializing AssessmentEvaluator with basic text matching")
        
    def precision_at_k(self, assessments: List[Dict[str, Any]], 
                      expected_keywords: List[str], k: int = 5) -> float:
        """
        Calculate precision@k - the proportion of top-k assessments matching any expected keyword.
        
        Args:
            assessments: List of assessment dictionaries
            expected_keywords: List of expected keywords
            k: Number of top assessments to consider
            
        Returns:
            Precision score (0-1)
        """
        if not assessments or not expected_keywords:
            return 0.0
            
        matched_count = 0
        top_k = assessments[:min(k, len(assessments))]
        
        if not top_k:
            return 0.0
            
        for assessment in top_k:
            description = assessment.get("description", "").lower()
            job_levels = assessment.get("job_levels", "").lower()
            languages = assessment.get("languages", "").lower()
            
            # Check if any keyword is in description or other fields
            if any(keyword.lower() in description or 
                   keyword.lower() in job_levels or 
                   keyword.lower() in languages 
                   for keyword in expected_keywords):
                matched_count += 1
                
        return matched_count / len(top_k)
    
    def recall_at_k(self, assessments: List[Dict[str, Any]], 
                   expected_keywords: List[str], k: int = 5) -> float:
        """
        Calculate recall@k - the proportion of expected keywords found in the top-k assessments.
        
        Args:
            assessments: List of assessment dictionaries
            expected_keywords: List of expected keywords
            k: Number of top assessments to consider
            
        Returns:
            Recall score (0-1)
        """
        if not assessments or not expected_keywords:
            return 0.0
            
        found_keywords = set()
        top_k = assessments[:min(k, len(assessments))]
        
        for keyword in expected_keywords:
            keyword_lower = keyword.lower()
            for assessment in top_k:
                description = assessment.get("description", "").lower()
                job_levels = assessment.get("job_levels", "").lower()
                languages = assessment.get("languages", "").lower()
                
                if (keyword_lower in description or 
                    keyword_lower in job_levels or 
                    keyword_lower in languages):
                    found_keywords.add(keyword_lower)
                    break
                    
        return len(found_keywords) / len(expected_keywords)
    
    def safe_cosine_similarity(self, vec1: Union[np.ndarray, list], 
                              vec2: Union[np.ndarray, list]) -> float:
        """
        Calculate cosine similarity between two vectors, handling different dimensions.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        # Convert to numpy arrays if they aren't already
        vec1_array = np.array(vec1)
        vec2_array = np.array(vec2)
        
        # Check if vectors are empty
        if vec1_array.size == 0 or vec2_array.size == 0:
            return 0.0
        
        # If dimensions don't match, use Jaccard similarity as fallback
        if vec1_array.shape != vec2_array.shape:
            try:
                # Convert to sets for Jaccard similarity
                set1 = set(np.where(vec1_array > 0)[0])
                set2 = set(np.where(vec2_array > 0)[0])
                
                # Calculate Jaccard similarity
                if not set1 and not set2:
                    return 0.0
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                return intersection / union if union > 0 else 0.0
            except Exception as e:
                logger.warning(f"Fallback similarity calculation failed: {e}")
                return 0.0
        
        # Regular cosine similarity
        norm_product = np.linalg.norm(vec1_array) * np.linalg.norm(vec2_array)
        if norm_product == 0:
            return 0.0
        return np.dot(vec1_array, vec2_array) / norm_product
    
    def text_to_vector(self, text: str) -> np.ndarray:
        """
        Convert text to a simple bag-of-words vector.
        
        Args:
            text: Input text
            
        Returns:
            Bag-of-words vector
        """
        if not text:
            return np.array([])
            
        # Simple word frequency
        words = text.lower().split()
        unique_words = sorted(set(words))
        word_to_idx = {word: i for i, word in enumerate(unique_words)}
        
        vector = np.zeros(len(word_to_idx))
        for word in words:
            if word in word_to_idx:
                vector[word_to_idx[word]] += 1
                
        return vector
    
    def mean_embedding_similarity(self, input_text: str, 
                                 assessments: List[Dict[str, Any]]) -> float:
        """
        Calculate mean semantic similarity between input text and assessments.
        
        Args:
            input_text: Input job description text
            assessments: List of assessment dictionaries
            
        Returns:
            Mean similarity score (0-1)
        """
        if not assessments or not input_text:
            return 0.0
            
        # Use basic text similarity with vector approach
        input_vec = self.text_to_vector(input_text)
        
        similarities = []
        for assessment in assessments:
            description = assessment.get("description", "")
            if description:
                desc_vec = self.text_to_vector(description)
                try:
                    similarity = self.safe_cosine_similarity(input_vec, desc_vec)
                    similarities.append(similarity)
                except Exception as e:
                    logger.warning(f"Similarity calculation failed: {e}")
                    similarities.append(0.0)
                
        return np.mean(similarities) if similarities else 0.0
    
    def f1_score(self, precision: float, recall: float) -> float:
        """
        Calculate F1 score from precision and recall.
        
        Args:
            precision: Precision score
            recall: Recall score
            
        Returns:
            F1 score (0-1)
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def evaluate_single_test(self, api_url: str, test_case: Dict[str, Any], 
                            k: int = 5, max_retries: int = 3, timeout: int = 60) -> Dict[str, float]:
        """
        Evaluate a single test case against the API.
        
        Args:
            api_url: URL of the assessment API endpoint
            test_case: Dictionary containing test case data
            k: Number of top assessments to consider
            max_retries: Maximum number of retry attempts for API calls
            timeout: Timeout in seconds for API calls
            
        Returns:
            Dictionary of evaluation metrics
        """
        input_text = test_case["input"]
        expected_keywords = test_case["expected_keywords"]
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Build the request payload
                data = json.dumps({"job_description": input_text, "top_n": k}).encode('utf-8')
                
                # Create the request
                req = urllib.request.Request(
                    api_url,
                    data=data,
                    headers={'Content-Type': 'application/json'}
                )
                
                logger.info(f"Sending request to {api_url}")
                
                # Send the request and get the response with increased timeout
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    response_data = response.read().decode('utf-8')
                    data = json.loads(response_data)
                    assessments = data.get("assessments", [])
                    
                    logger.info(f"Received {len(assessments)} assessments from API")
                
                precision = self.precision_at_k(assessments, expected_keywords, k)
                recall = self.recall_at_k(assessments, expected_keywords, k)
                f1 = self.f1_score(precision, recall)
                
                # Try to calculate similarity, but handle errors gracefully
                try:
                    similarity = self.mean_embedding_similarity(input_text, assessments)
                except Exception as e:
                    logger.error(f"Similarity calculation failed: {e}")
                    similarity = 0.0
                
                return {
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "semantic_similarity": similarity,
                    "num_assessments": len(assessments)
                }
                
            except urllib.error.URLError as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.warning(f"API request failed (attempt {retry_count}/{max_retries}): {e}")
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API request failed after {max_retries} attempts: {e}")
                    return {
                        "error": str(e),
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1_score": 0.0,
                        "semantic_similarity": 0.0,
                        "num_assessments": 0
                    }
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse API response: {e}")
                return {
                    "error": f"JSON Parse Error: {str(e)}",
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "semantic_similarity": 0.0,
                    "num_assessments": 0
                }
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    return {
                        "error": f"Unexpected error: {str(e)}",
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1_score": 0.0,
                        "semantic_similarity": 0.0,
                        "num_assessments": 0
                    }
    
    def evaluate_test_set(self, api_url: str, test_set: List[Dict[str, Any]], 
                         k: int = 5) -> List[Dict[str, Any]]:
        """
        Evaluate all test cases and return detailed results.
        
        Args:
            api_url: URL of the assessment API endpoint
            test_set: List of test case dictionaries
            k: Number of top assessments to consider
            
        Returns:
            List of test cases with evaluation metrics
        """
        results = []
        
        for i, test_case in enumerate(test_set):
            logger.info(f"Evaluating test case {i+1}/{len(test_set)}")
            metrics = self.evaluate_single_test(api_url, test_case, k)
            
            results.append({
                "test_case": test_case,
                "metrics": metrics
            })
            
            # Add a small delay between requests to avoid overloading the hosted API
            if i < len(test_set) - 1:
                time.sleep(1)
            
        return results
    
    def print_evaluation_summary(self, results: List[Dict[str, Any]], k: int = 5) -> None:
        """
        Print a summary of evaluation results.
        
        Args:
            results: List of evaluation result dictionaries
            k: Number of top assessments considered
        """
        if not results:
            logger.warning("No evaluation results to summarize")
            return
            
        print("\n===== EVALUATION SUMMARY =====")
        
        avg_metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "semantic_similarity": 0.0
        }
        
        for i, result in enumerate(results):
            test_case = result["test_case"]
            metrics = result["metrics"]
            
            print(f"\nTest Case {i+1}:")
            print(f"Input: {test_case['input']}")
            print(f"Expected Keywords: {', '.join(test_case['expected_keywords'])}")
            
            # Check for errors in this test case
            if "error" in metrics:
                print(f"❌ Error: {metrics['error']}")
                continue
                
            print(f"Precision@{k}: {metrics['precision']:.4f}")
            print(f"Recall@{k}: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"Semantic Similarity: {metrics['semantic_similarity']:.4f}")
            print(f"Number of assessments returned: {metrics['num_assessments']}")
            
            for key in avg_metrics:
                avg_metrics[key] += metrics.get(key, 0.0)
        
        # Count successful test cases
        successful_tests = sum(1 for r in results if "error" not in r["metrics"])
        
        if successful_tests > 0:
            # Calculate averages
            for key in avg_metrics:
                avg_metrics[key] /= successful_tests
                
            print("\n===== AVERAGE METRICS =====")
            for key, value in avg_metrics.items():
                print(f"Average {key.replace('_', ' ').title()}: {value:.4f}")
        else:
            print("\n❌ No successful test cases to calculate average metrics")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate assessment recommendation API")
    parser.add_argument("--api-url", type=str, required=True, 
                        help="URL of the assessment API endpoint")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top assessments to consider")
    args = parser.parse_args()
    
    # Test cases - expanded with relevant keywords for assessment matching
    TEST_SET = [
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
    
    # Run evaluation
    try:
        print("\n🔄 Starting evaluation...")
        print(f"🌐 API URL: {args.api_url}")
        print(f"🔢 Test cases: {len(TEST_SET)}")
        print(f"🏆 Top-k: {args.top_k}")
        
        evaluator = AssessmentEvaluator()
        results = evaluator.evaluate_test_set(args.api_url, TEST_SET, args.top_k)
        evaluator.print_evaluation_summary(results, args.top_k)
        
        print("\n✅ Evaluation completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n\nTroubleshooting tips:")
        print("1. Make sure the API server is running at:", args.api_url)
        print("2. Check that the API accepts POST requests with this format:")
        print("   {\"job_description\": \"your text here\", \"top_n\": 5}")
        print("3. Ensure the API returns JSON with an 'assessments' array")
        print("4. For hosted APIs, check if there are rate limits or if the service is active")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Standalone test of evaluation metrics (no external dependencies)."""

# Import paths fixed - sys.path manipulation removed
import sys
import os
))

# Import just the evaluation metrics module directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "evaluation_metrics",
    "src/use_cases/custom_prompts/evaluation_metrics.py"
)
evaluation_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation_metrics)

# Import classes from the module
ResponseLengthMetric = evaluation_metrics.ResponseLengthMetric
SentimentMetric = evaluation_metrics.SentimentMetric
CoherenceMetric = evaluation_metrics.CoherenceMetric
ResponseDiversityMetric = evaluation_metrics.ResponseDiversityMetric
CustomMetric = evaluation_metrics.CustomMetric
MetricSuite = evaluation_metrics.MetricSuite
evaluate_response = evaluation_metrics.evaluate_response
evaluate_responses = evaluation_metrics.evaluate_responses

print("Testing Evaluation Metrics (Standalone)")
print("=" * 60)

# Test responses
response1 = """This is a great example of a helpful response! I'm happy to assist you. 
The solution works perfectly and provides excellent results. 
Thank you for asking such a wonderful question."""

response2 = """Unfortunately, this approach has several problems and issues. 
The results are terrible and the implementation failed completely. 
This is the worst possible outcome."""

response3 = """The situation is okay. It's an average solution with normal results. 
The implementation is adequate and acceptable for standard use cases."""

response4 = """To solve this problem, we need to consider multiple factors. 
First, we analyze the requirements. Then, we implement the solution. 
Finally, we test and validate the results. This ensures a comprehensive approach."""

# Test 1: Response Length Metric
print("\n1. Response Length Metric")
print("-" * 40)
length_metric = ResponseLengthMetric()
result = length_metric.calculate(response1)
print(f"Response 1 length metrics: {result.value}")

# Test 2: Sentiment Metric
print("\n2. Sentiment Analysis")
print("-" * 40)
sentiment_metric = SentimentMetric()
for i, response in enumerate([response1, response2, response3], 1):
    result = sentiment_metric.calculate(response)
    print(f"Response {i} sentiment: {result.value['label']} (score: {result.value['score']})")

# Test 3: Coherence Metric
print("\n3. Coherence Analysis")
print("-" * 40)
coherence_metric = CoherenceMetric()
result = coherence_metric.calculate(response4)
print(f"Response 4 coherence: {result.value}")

# Test 4: Response Diversity
print("\n4. Response Diversity (Multiple Responses)")
print("-" * 40)
diversity_metric = ResponseDiversityMetric()
responses = [response1, response2, response3, response4]
diversity_results = diversity_metric.calculate_batch(responses)
print(f"Diversity across 4 responses: {diversity_results[0].value}")

# Test 5: Custom Metric
print("\n5. Custom Metric Example")
print("-" * 40)
def question_count(response, **kwargs):
    """Count the number of questions in the response."""
    return len([s for s in response.split('.') if '?' in s])

question_metric = CustomMetric("question_count", question_count)
test_response = "How are you? What can I help with? I'm here to assist. Do you need anything?"
result = question_metric.calculate(test_response)
print(f"Questions in response: {result.value}")

# Test 6: Metric Suite
print("\n6. Metric Suite (All Default Metrics)")
print("-" * 40)
results = evaluate_response(response1)
for metric_name, metric_data in results.items():
    print(f"{metric_name}: {metric_data['value']}")

# Test 7: Batch Evaluation with Aggregation
print("\n7. Batch Evaluation with Aggregation")
print("-" * 40)
batch_results = evaluate_responses([response1, response2, response3, response4])
print("Aggregated results:")
for metric_name, aggregated in batch_results['aggregated'].items():
    if metric_name == 'diversity':
        print(f"  {metric_name}: diversity_score={aggregated['value']['diversity_score']}")
    else:
        print(f"  {metric_name}: {aggregated}")

# Test 8: Edge Cases
print("\n8. Edge Case Testing")
print("-" * 40)
# Empty response
empty_result = evaluate_response("")
print(f"Empty response - length: {empty_result['response_length']['value']['characters']}")

# Single word
single_word_result = evaluate_response("Hello")
print(f"Single word - coherence: {single_word_result['coherence']['value']['score']}")

# Very repetitive text
repetitive = "test " * 50
repetitive_result = coherence_metric.calculate(repetitive)
print(f"Repetitive text - repetition score: {repetitive_result.value['repetition_score']}")

print("\n✅ All evaluation metric tests completed successfully!")
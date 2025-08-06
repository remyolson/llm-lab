"""
A/B Testing API endpoints for model comparison
"""

import asyncio
import logging
import random
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from .models import ABTest, ABTestConfig, APIResponse, ListResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage (replace with database)
ab_tests_db: Dict[str, ABTest] = {}


@router.post("/", response_model=APIResponse)
async def create_ab_test(test_config: ABTestConfig, background_tasks: BackgroundTasks):
    """Create a new A/B test"""

    test_id = str(uuid.uuid4())

    test = ABTest(
        id=test_id,
        name=test_config.name,
        description=test_config.description,
        baselineModel=test_config.baselineModel,
        candidateModel=test_config.candidateModel,
        status="preparing",
        createdAt=datetime.utcnow(),
    )

    ab_tests_db[test_id] = test

    # Start A/B test in background
    background_tasks.add_task(run_ab_test, test_id, test_config)

    logger.info(f"Created A/B test {test_id}: {test_config.name}")

    return APIResponse(success=True, message="A/B test created successfully", data={"id": test_id})


@router.get("/", response_model=ListResponse)
async def list_ab_tests(page: int = 1, page_size: int = 20, status: Optional[str] = None):
    """List all A/B tests with pagination"""

    # Filter by status if provided
    filtered_tests = list(ab_tests_db.values())
    if status:
        filtered_tests = [test for test in filtered_tests if test.status == status]

    # Sort by creation date (newest first)
    filtered_tests.sort(key=lambda x: x.createdAt, reverse=True)

    # Pagination
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = filtered_tests[start_idx:end_idx]

    return ListResponse(items=paginated, total=len(filtered_tests), page=page, pageSize=page_size)


@router.get("/{test_id}", response_model=ABTest)
async def get_ab_test(test_id: str):
    """Get A/B test details"""

    if test_id not in ab_tests_db:
        raise HTTPException(status_code=404, detail="A/B test not found")

    return ab_tests_db[test_id]


@router.delete("/{test_id}", response_model=APIResponse)
async def delete_ab_test(test_id: str):
    """Delete an A/B test"""

    if test_id not in ab_tests_db:
        raise HTTPException(status_code=404, detail="A/B test not found")

    test = ab_tests_db[test_id]

    # Stop test if it's running
    if test.status == "running":
        test.status = "stopped"

    del ab_tests_db[test_id]

    logger.info(f"Deleted A/B test {test_id}")

    return APIResponse(success=True, message="A/B test deleted successfully")


@router.post("/{test_id}/start", response_model=APIResponse)
async def start_ab_test(test_id: str, background_tasks: BackgroundTasks):
    """Start an A/B test"""

    if test_id not in ab_tests_db:
        raise HTTPException(status_code=404, detail="A/B test not found")

    test = ab_tests_db[test_id]

    if test.status == "running":
        raise HTTPException(status_code=400, detail="A/B test is already running")

    test.status = "running"
    test.startedAt = datetime.utcnow()

    logger.info(f"Started A/B test {test_id}")

    return APIResponse(success=True, message="A/B test started successfully")


@router.post("/{test_id}/stop", response_model=APIResponse)
async def stop_ab_test(test_id: str):
    """Stop a running A/B test"""

    if test_id not in ab_tests_db:
        raise HTTPException(status_code=404, detail="A/B test not found")

    test = ab_tests_db[test_id]

    if test.status != "running":
        raise HTTPException(status_code=400, detail="A/B test is not running")

    test.status = "stopped"
    test.completedAt = datetime.utcnow()

    logger.info(f"Stopped A/B test {test_id}")

    return APIResponse(success=True, message="A/B test stopped successfully")


@router.get("/{test_id}/results")
async def get_ab_test_results(test_id: str):
    """Get A/B test results"""

    if test_id not in ab_tests_db:
        raise HTTPException(status_code=404, detail="A/B test not found")

    test = ab_tests_db[test_id]

    if not test.results:
        # Generate sample results if not available
        test.results = generate_sample_results()

    return test.results


@router.post("/{test_id}/compare")
async def compare_models(test_id: str, prompt: str, num_samples: int = 1):
    """Compare model responses for a given prompt"""

    if test_id not in ab_tests_db:
        raise HTTPException(status_code=404, detail="A/B test not found")

    test = ab_tests_db[test_id]

    # Simulate model responses
    baseline_responses = []
    candidate_responses = []

    for i in range(num_samples):
        await asyncio.sleep(0.1)  # Simulate inference time

        baseline_response = f"Baseline model response {i + 1} to: {prompt}"
        candidate_response = f"Candidate model response {i + 1} to: {prompt}"

        baseline_responses.append(
            {
                "response": baseline_response,
                "latency_ms": random.randint(200, 500),
                "tokens": len(baseline_response.split()),
                "quality_score": random.uniform(0.7, 0.9),
            }
        )

        candidate_responses.append(
            {
                "response": candidate_response,
                "latency_ms": random.randint(180, 450),
                "tokens": len(candidate_response.split()),
                "quality_score": random.uniform(0.75, 0.95),
            }
        )

    return {
        "test_id": test_id,
        "prompt": prompt,
        "baseline_model": test.baselineModel,
        "candidate_model": test.candidateModel,
        "baseline_responses": baseline_responses,
        "candidate_responses": candidate_responses,
        "comparison_summary": {
            "avg_baseline_latency": sum(r["latency_ms"] for r in baseline_responses)
            / len(baseline_responses),
            "avg_candidate_latency": sum(r["latency_ms"] for r in candidate_responses)
            / len(candidate_responses),
            "avg_baseline_quality": sum(r["quality_score"] for r in baseline_responses)
            / len(baseline_responses),
            "avg_candidate_quality": sum(r["quality_score"] for r in candidate_responses)
            / len(candidate_responses),
        },
    }


@router.get("/{test_id}/metrics")
async def get_ab_test_metrics(test_id: str):
    """Get detailed metrics for an A/B test"""

    if test_id not in ab_tests_db:
        raise HTTPException(status_code=404, detail="A/B test not found")

    test = ab_tests_db[test_id]

    # Generate sample metrics
    metrics = {
        "traffic_distribution": {"baseline": 0.5, "candidate": 0.5},
        "performance_metrics": {
            "baseline": {
                "avg_latency_ms": 350,
                "p95_latency_ms": 580,
                "throughput_rps": 45,
                "error_rate": 0.02,
            },
            "candidate": {
                "avg_latency_ms": 320,
                "p95_latency_ms": 520,
                "throughput_rps": 48,
                "error_rate": 0.015,
            },
        },
        "quality_metrics": {
            "baseline": {
                "avg_quality_score": 0.82,
                "user_satisfaction": 0.78,
                "task_completion_rate": 0.85,
            },
            "candidate": {
                "avg_quality_score": 0.87,
                "user_satisfaction": 0.83,
                "task_completion_rate": 0.89,
            },
        },
        "statistical_significance": {
            "latency_improvement": {"significant": True, "p_value": 0.023, "confidence": 0.95},
            "quality_improvement": {"significant": True, "p_value": 0.011, "confidence": 0.95},
        },
        "sample_size": {"baseline": 1250, "candidate": 1280, "total": 2530},
    }

    return metrics


# Background task functions


async def run_ab_test(test_id: str, config: ABTestConfig):
    """Run A/B test in background"""
    try:
        if test_id not in ab_tests_db:
            return

        test = ab_tests_db[test_id]

        # Simulate test preparation
        test.status = "preparing"
        await asyncio.sleep(3)

        # Start the test
        test.status = "running"
        test.startedAt = datetime.utcnow()

        # Simulate test running (in reality, this would collect real data)
        await asyncio.sleep(10)  # Simulate some test duration

        # Generate results
        test.results = generate_sample_results()
        test.status = "completed"
        test.completedAt = datetime.utcnow()

        logger.info(f"A/B test {test_id} completed")

    except Exception as e:
        logger.error(f"Error running A/B test {test_id}: {e}")
        if test_id in ab_tests_db:
            test = ab_tests_db[test_id]
            test.status = "failed"


def generate_sample_results() -> Dict[str, Any]:
    """Generate sample A/B test results"""

    return {
        "winner": "candidate",
        "confidence": 0.95,
        "improvement": {
            "latency": -8.5,  # 8.5% improvement (negative = better)
            "quality": 6.1,  # 6.1% improvement
            "user_satisfaction": 6.4,  # 6.4% improvement
        },
        "metrics": {
            "baseline": {
                "latency_ms": 350,
                "quality_score": 0.82,
                "user_satisfaction": 0.78,
                "samples": 1250,
            },
            "candidate": {
                "latency_ms": 320,
                "quality_score": 0.87,
                "user_satisfaction": 0.83,
                "samples": 1280,
            },
        },
        "statistical_tests": {
            "latency_t_test": {"p_value": 0.023, "significant": True},
            "quality_t_test": {"p_value": 0.011, "significant": True},
        },
        "recommendation": "Deploy candidate model - shows significant improvement in both latency and quality",
    }

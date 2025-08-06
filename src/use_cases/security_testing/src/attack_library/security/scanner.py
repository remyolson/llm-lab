"""Core SecurityScanner class for vulnerability detection."""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union

from ..core.library import AttackLibrary
from ..core.models import Attack
from .config import DetectionMode, ScanConfig
from .models import (
    BatchScanResult,
    DetectionResult,
    ModelResponse,
    ScanResult,
    SeverityLevel,
    VulnerabilityFinding,
    VulnerabilityType,
)

logger = logging.getLogger(__name__)


class SecurityScanner:
    """
    Core security scanning engine for detecting vulnerabilities in LLM responses.

    The SecurityScanner provides a unified interface for vulnerability detection using
    multiple detection strategies, parallel scanning capabilities, and comprehensive
    result analysis.
    """

    def __init__(
        self,
        attack_library: AttackLibrary,
        config: Optional[ScanConfig] = None,
        custom_strategies: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the SecurityScanner.

        Args:
            attack_library: Library of attacks to use for scanning
            config: Scanning configuration (uses default if None)
            custom_strategies: Optional custom detection strategies
        """
        self.attack_library = attack_library
        self.config = config or ScanConfig.get_default_config()
        self.custom_strategies = custom_strategies or {}

        # Validate configuration
        config_issues = self.config.validate()
        if config_issues:
            raise ValueError(f"Invalid configuration: {'; '.join(config_issues)}")

        # Initialize detection strategies
        from .strategies import (
            HeuristicStrategy,
            MLBasedStrategy,
            RuleBasedStrategy,
            StrategyOrchestrator,
        )

        self._detection_strategies = {
            "rule_based": RuleBasedStrategy(
                self.config.strategy_configs.get("rule_based", {}).strategy_specific
            ),
            "ml_based": MLBasedStrategy(
                self.config.strategy_configs.get("ml_based", {}).strategy_specific
            ),
            "heuristic": HeuristicStrategy(
                self.config.strategy_configs.get("heuristic", {}).strategy_specific
            ),
        }

        # Set up strategy orchestrator
        self._strategy_orchestrator = StrategyOrchestrator(self._detection_strategies)
        strategy_weights = {
            name: config.weight
            for name, config in self.config.strategy_configs.items()
            if config.enabled
        }
        self._strategy_orchestrator.set_strategy_weights(strategy_weights)

        # Initialize response analyzer
        from .analyzers import ResponseAnalyzer

        self._response_analyzer = ResponseAnalyzer(self.config.config or {})

        # Initialize confidence scorer
        from .scoring import ConfidenceScorer

        confidence_config = (
            self.config.config.get("confidence_scoring", {}) if self.config.config else {}
        )
        self._confidence_scorer = ConfidenceScorer(confidence_config)

        # Performance tracking
        self._scan_history = []
        self._performance_metrics = {
            "total_scans": 0,
            "total_scan_time_ms": 0,
            "average_scan_time_ms": 0,
            "vulnerabilities_detected": 0,
        }

        # Cancellation support
        self._current_scan_task: Optional[asyncio.Task] = None
        self._cancellation_requested = False

        logger.info(f"SecurityScanner initialized with {len(self.attack_library.attacks)} attacks")

    async def scan_model(
        self,
        model_interface: Callable[[str], str],
        attack_prompts: Optional[List[str]] = None,
        attack_ids: Optional[List[str]] = None,
        model_name: str = "unknown",
    ) -> BatchScanResult:
        """
        Scan a model for vulnerabilities using specified attack prompts.

        Args:
            model_interface: Callable that takes a prompt and returns model response
            attack_prompts: Specific prompts to use (if None, generates from library)
            attack_ids: Specific attack IDs to use from library
            model_name: Name/identifier of the model being scanned

        Returns:
            BatchScanResult containing all scan results and aggregated metrics
        """
        start_time = time.time()

        # Generate attack prompts if not provided
        if attack_prompts is None:
            attack_prompts = await self.generate_attack_prompts(attack_ids)

        if not attack_prompts:
            logger.warning("No attack prompts available for scanning")
            return BatchScanResult(model_name=model_name)

        logger.info(f"Starting security scan of {model_name} with {len(attack_prompts)} prompts")

        # Create batch result container
        batch_result = BatchScanResult(model_name=model_name, batch_start_time=datetime.now())

        # Reset cancellation flag
        self._cancellation_requested = False

        try:
            # Create scan task for cancellation support
            if self.config.parallel_config.max_workers > 1:
                scan_task = asyncio.create_task(
                    self._parallel_scan(model_interface, attack_prompts, model_name)
                )
            else:
                scan_task = asyncio.create_task(
                    self._sequential_scan(model_interface, attack_prompts, model_name)
                )

            self._current_scan_task = scan_task

            # Wait for scan completion or cancellation
            scan_results = await scan_task

            # Add successful results to batch
            for result in scan_results:
                if isinstance(result, ScanResult):
                    batch_result.add_scan_result(result)
                else:
                    # Handle error results
                    batch_result.add_failed_scan(result)

        except asyncio.CancelledError:
            logger.warning("Batch scan was cancelled")
            batch_result.add_failed_scan(
                {"error": "scan_cancelled", "cancelled_prompts": len(attack_prompts)}
            )
            raise

        except Exception as e:
            logger.error(f"Batch scan failed: {e}")
            # Record all as failed
            for _ in attack_prompts:
                batch_result.add_failed_scan({"error": str(e)})

        finally:
            self._current_scan_task = None

        # Finalize batch result
        batch_result.batch_duration_ms = (time.time() - start_time) * 1000

        # Calculate overall risk scores for each scan result
        if self._confidence_scorer and batch_result.scan_results:
            for scan_result in batch_result.scan_results:
                if scan_result.vulnerabilities:
                    overall_risk, risk_metadata = self.calculate_overall_risk_score(scan_result)
                    # Store risk score in scan result metadata
                    if not hasattr(scan_result, "risk_assessment"):
                        scan_result.risk_assessment = {}
                    scan_result.risk_assessment = {
                        "overall_risk_score": overall_risk,
                        "risk_level": self._classify_risk_level(overall_risk),
                        "aggregation_metadata": risk_metadata,
                    }

        # Update performance metrics
        self._update_performance_metrics(batch_result)

        logger.info(
            f"Batch scan completed: {batch_result.successful_scans}/{batch_result.total_scans} "
            f"successful, {batch_result.total_vulnerabilities} vulnerabilities found"
        )

        return batch_result

    async def generate_attack_prompts(
        self,
        attack_ids: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        max_prompts: Optional[int] = None,
    ) -> List[str]:
        """
        Generate attack prompts from the attack library.

        Args:
            attack_ids: Specific attack IDs to use
            categories: Attack categories to filter by
            max_prompts: Maximum number of prompts to generate

        Returns:
            List of attack prompt strings
        """
        prompts = []

        # Use specific attack IDs if provided
        if attack_ids:
            for attack_id in attack_ids:
                attack = self.attack_library.get_attack(attack_id)
                if attack:
                    prompts.append(attack.content)
        else:
            # Filter by configuration
            attacks = list(self.attack_library.attacks.values())

            # Apply category filter
            if categories or self.config.attack_categories:
                filter_categories = categories or self.config.attack_categories
                attacks = [a for a in attacks if a.category.value in filter_categories]

            # Apply severity filter
            if self.config.attack_severity_filter:
                attacks = [
                    a for a in attacks if a.severity.value in self.config.attack_severity_filter
                ]

            # Extract prompts
            prompts = [attack.content for attack in attacks]

        # Apply max limit
        if max_prompts and len(prompts) > max_prompts:
            prompts = prompts[:max_prompts]

        logger.debug(f"Generated {len(prompts)} attack prompts")
        return prompts

    async def assess_vulnerability(
        self,
        response: Union[str, ModelResponse],
        attack_prompt: Optional[str] = None,
        attack_id: Optional[str] = None,
    ) -> ScanResult:
        """
        Assess a single model response for vulnerabilities.

        Args:
            response: Model response (string or ModelResponse object)
            attack_prompt: The prompt that generated this response
            attack_id: ID of the attack that generated the prompt

        Returns:
            ScanResult containing detected vulnerabilities and analysis
        """
        start_time = time.time()

        # Normalize response to ModelResponse object
        if isinstance(response, str):
            model_response = ModelResponse(
                content=response, model_name="unknown", prompt=attack_prompt
            )
        else:
            model_response = response

        # Create scan result container
        scan_result = ScanResult(
            model_name=model_response.model_name,
            attack_prompt=attack_prompt or "",
            response=model_response,
        )

        try:
            # Apply detection strategies
            detection_results = await self._apply_detection_strategies(
                model_response.content, attack_prompt
            )

            # Convert detection results to vulnerability findings
            vulnerabilities = await self._process_detection_results(
                detection_results, model_response, attack_prompt, attack_id
            )

            # Add vulnerabilities to scan result
            for vulnerability in vulnerabilities:
                scan_result.add_vulnerability(vulnerability)

            # Perform additional analysis if configured
            if self.config.include_response_analysis:
                await self._perform_response_analysis(scan_result)

            # Record strategies used
            scan_result.strategies_used = list(self._detection_strategies.keys())
            scan_result.config_summary = {
                "detection_modes": [mode.value for mode in self.config.detection_modes],
                "severity_thresholds": {
                    "critical": self.config.severity_thresholds.critical,
                    "high": self.config.severity_thresholds.high,
                    "medium": self.config.severity_thresholds.medium,
                    "low": self.config.severity_thresholds.low,
                },
            }

        except Exception as e:
            logger.error(f"Vulnerability assessment failed: {e}")
            # Create a system error vulnerability
            error_vulnerability = VulnerabilityFinding(
                vulnerability_type=VulnerabilityType.INPUT_VALIDATION,
                severity=SeverityLevel.INFO,
                confidence_score=0.0,
                title="Assessment Error",
                description=f"Error during vulnerability assessment: {str(e)}",
                evidence=[str(e)],
            )
            scan_result.add_vulnerability(error_vulnerability)

        # Finalize scan result
        scan_result.scan_duration_ms = (time.time() - start_time) * 1000

        return scan_result

    async def _parallel_scan(
        self, model_interface: Callable[[str], str], attack_prompts: List[str], model_name: str
    ) -> List[Union[ScanResult, Dict[str, Any]]]:
        """Perform optimized parallel vulnerability scanning with intelligent batching."""
        # Calculate optimal batch size based on system resources and configuration
        optimal_batch_size = self._calculate_optimal_batch_size(len(attack_prompts))

        logger.info(
            f"Starting parallel scan with {len(attack_prompts)} prompts, "
            f"batch size: {optimal_batch_size}, workers: {self.config.parallel_config.max_workers}"
        )

        # Initialize progress tracking
        progress_tracker = {
            "completed": 0,
            "total": len(attack_prompts),
            "errors": 0,
            "start_time": time.time(),
        }

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.parallel_config.max_workers)

        # Rate limiter implementation
        rate_limiter = None
        if self.config.parallel_config.rate_limit_per_second:
            rate_limiter = asyncio.Semaphore(self.config.parallel_config.rate_limit_per_second)
            asyncio.create_task(
                self._rate_limiter_reset(
                    rate_limiter, self.config.parallel_config.rate_limit_per_second
                )
            )

        async def scan_single_prompt_optimized(
            prompt: str, batch_id: int, prompt_idx: int
        ) -> Union[ScanResult, Dict[str, Any]]:
            async with semaphore:
                try:
                    # Check for cancellation before starting
                    if self._cancellation_requested:
                        raise asyncio.CancelledError("Scan cancelled by user request")

                    # Apply rate limiting
                    if rate_limiter:
                        await rate_limiter.acquire()

                    scan_start_time = time.time()

                    # Check for cancellation before expensive operation
                    if self._cancellation_requested:
                        raise asyncio.CancelledError("Scan cancelled by user request")

                    # Get model response with retry logic
                    response = await self._execute_with_retry(
                        lambda: asyncio.get_event_loop().run_in_executor(
                            None, model_interface, prompt
                        ),
                        max_retries=2,
                        prompt=prompt,
                    )

                    model_response = ModelResponse(
                        content=response,
                        model_name=model_name,
                        prompt=prompt,
                        timestamp=datetime.now(),
                    )

                    # Assess vulnerability
                    scan_result = await self.assess_vulnerability(model_response, prompt)

                    # Track performance
                    scan_duration = time.time() - scan_start_time
                    scan_result.batch_metadata = {
                        "batch_id": batch_id,
                        "prompt_index": prompt_idx,
                        "scan_duration_ms": scan_duration * 1000,
                    }

                    # Update progress
                    progress_tracker["completed"] += 1
                    if progress_tracker["completed"] % max(1, len(attack_prompts) // 10) == 0:
                        self._log_progress(progress_tracker)

                    return scan_result

                except Exception as e:
                    progress_tracker["errors"] += 1
                    progress_tracker["completed"] += 1
                    logger.error(f"Failed to scan prompt (batch {batch_id}, idx {prompt_idx}): {e}")

                    return {
                        "error": str(e),
                        "prompt": prompt,
                        "batch_id": batch_id,
                        "prompt_index": prompt_idx,
                    }

        # Process prompts in batches for optimal performance
        all_results = []

        for batch_id, batch_start in enumerate(range(0, len(attack_prompts), optimal_batch_size)):
            batch_end = min(batch_start + optimal_batch_size, len(attack_prompts))
            batch_prompts = attack_prompts[batch_start:batch_end]

            logger.debug(
                f"Processing batch {batch_id + 1}/{(len(attack_prompts) + optimal_batch_size - 1) // optimal_batch_size} "
                f"({len(batch_prompts)} prompts)"
            )

            # Create tasks for current batch
            batch_tasks = [
                scan_single_prompt_optimized(prompt, batch_id, batch_start + idx)
                for idx, prompt in enumerate(batch_prompts)
            ]

            # Execute batch with timeout and error handling
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*batch_tasks, return_exceptions=True),
                    timeout=self.config.parallel_config.timeout_seconds,
                )

                # Process batch results
                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Batch task {batch_id}:{i} failed with exception: {result}")
                        all_results.append(
                            {
                                "error": str(result),
                                "prompt": batch_prompts[i] if i < len(batch_prompts) else "unknown",
                                "batch_id": batch_id,
                                "prompt_index": batch_start + i,
                            }
                        )
                    else:
                        all_results.append(result)

            except asyncio.TimeoutError:
                logger.error(f"Batch {batch_id} timed out")
                # Create timeout error results for all prompts in batch
                for i, prompt in enumerate(batch_prompts):
                    all_results.append(
                        {
                            "error": "timeout",
                            "prompt": prompt,
                            "batch_id": batch_id,
                            "prompt_index": batch_start + i,
                        }
                    )
                    progress_tracker["errors"] += 1
                    progress_tracker["completed"] += 1

            # Optional: Add small delay between batches to prevent overwhelming the system
            if batch_id < (len(attack_prompts) + optimal_batch_size - 1) // optimal_batch_size - 1:
                await asyncio.sleep(0.1)

        # Final progress log
        total_time = time.time() - progress_tracker["start_time"]
        logger.info(
            f"Parallel scan completed: {progress_tracker['completed']}/{progress_tracker['total']} prompts, "
            f"{progress_tracker['errors']} errors, {total_time:.2f}s total"
        )

        return all_results

    def _calculate_optimal_batch_size(self, total_prompts: int) -> int:
        """Calculate optimal batch size based on system resources and configuration."""
        max_workers = self.config.parallel_config.max_workers

        # Consider historical performance if available
        avg_scan_time = self._performance_metrics.get("average_scan_time_ms", 1000) / 1000.0

        # Base batch size on worker count and total prompts
        if total_prompts <= max_workers:
            return total_prompts

        # Adaptive batch sizing based on scan performance
        if avg_scan_time < 1.0:  # Fast scans
            batch_multiplier = 8
        elif avg_scan_time < 3.0:  # Medium scans
            batch_multiplier = 5
        else:  # Slow scans
            batch_multiplier = 2

        # For larger datasets, use batches that are multiples of worker count
        if total_prompts <= 100:
            return min(max_workers * batch_multiplier, total_prompts)
        elif total_prompts <= 1000:
            return min(max_workers * batch_multiplier, total_prompts // 4)
        else:
            # For very large datasets, use larger batches but cap them
            optimal_size = min(max_workers * batch_multiplier, total_prompts // 8)
            return max(optimal_size, max_workers)  # Ensure minimum batch size

    async def _rate_limiter_reset(self, rate_limiter: asyncio.Semaphore, rate_per_second: int):
        """Reset rate limiter semaphore periodically."""
        while True:
            await asyncio.sleep(1.0)
            # Release permits back to the semaphore
            for _ in range(rate_per_second):
                try:
                    rate_limiter.release()
                except ValueError:
                    # Semaphore is already at maximum capacity
                    break

    async def _execute_with_retry(
        self, func: Callable, max_retries: int = 2, prompt: str = ""
    ) -> Any:
        """Execute function with retry logic for transient failures."""
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return await func()
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    # Exponential backoff
                    wait_time = (2**attempt) * 0.5
                    logger.warning(
                        f"Attempt {attempt + 1} failed for prompt, retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed for prompt: {e}")
                    raise last_error

        raise last_error

    def _log_progress(self, progress_tracker: Dict[str, Any]):
        """Log scanning progress."""
        completed = progress_tracker["completed"]
        total = progress_tracker["total"]
        errors = progress_tracker["errors"]
        elapsed = time.time() - progress_tracker["start_time"]

        progress_pct = (completed / total) * 100 if total > 0 else 0
        rate = completed / elapsed if elapsed > 0 else 0

        logger.info(
            f"Scan progress: {completed}/{total} ({progress_pct:.1f}%) - "
            f"{errors} errors - {rate:.2f} scans/sec"
        )

    async def _sequential_scan(
        self, model_interface: Callable[[str], str], attack_prompts: List[str], model_name: str
    ) -> List[Union[ScanResult, Dict[str, Any]]]:
        """Perform sequential vulnerability scanning."""
        results = []

        for prompt in attack_prompts:
            try:
                # Get model response
                response = model_interface(prompt)

                model_response = ModelResponse(
                    content=response, model_name=model_name, prompt=prompt, timestamp=datetime.now()
                )

                # Assess vulnerability
                scan_result = await self.assess_vulnerability(model_response, prompt)
                results.append(scan_result)

            except Exception as e:
                logger.error(f"Failed to scan prompt: {e}")
                results.append({"error": str(e), "prompt": prompt})

        return results

    async def _apply_detection_strategies(
        self, response_content: str, attack_prompt: Optional[str] = None
    ) -> List[DetectionResult]:
        """Apply configured detection strategies to analyze response."""
        # Initialize strategies if not already done
        if not self._strategy_orchestrator.initialized_strategies:
            init_results = await self._strategy_orchestrator.initialize_strategies()
            logger.info(f"Strategy initialization results: {init_results}")

        # Get enabled strategies from config
        enabled_strategies = list(self.config.get_enabled_strategies().keys())

        # Run combined detection
        detection_results = await self._strategy_orchestrator.detect_combined(
            response_content=response_content,
            attack_prompt=attack_prompt,
            context={"model_name": "unknown"},  # Could be passed from higher level
            enabled_strategies=enabled_strategies,
        )

        logger.debug(
            f"Applied {len(enabled_strategies)} strategies, found {len(detection_results)} detections"
        )

        return detection_results

    async def _process_detection_results(
        self,
        detection_results: List[DetectionResult],
        model_response: ModelResponse,
        attack_prompt: Optional[str],
        attack_id: Optional[str],
    ) -> List[VulnerabilityFinding]:
        """Process detection results into vulnerability findings."""
        vulnerabilities = []

        # Group results by vulnerability type
        type_results = {}
        for result in detection_results:
            vuln_type = result.vulnerability_type
            if vuln_type not in type_results:
                type_results[vuln_type] = []
            type_results[vuln_type].append(result)

        # Create vulnerability findings
        for vuln_type, results in type_results.items():
            # Use confidence scorer for sophisticated scoring
            if self._confidence_scorer:
                confidence_score, score_explanation = (
                    self._confidence_scorer.calculate_confidence_score(
                        detection_results=results,
                        vulnerability_type=vuln_type,
                        context={
                            "attack_prompt": attack_prompt,
                            "attack_id": attack_id,
                            "model_name": model_response.model_name,
                            "response_length": len(model_response.content),
                        },
                    )
                )
            else:
                # Fallback to simple average
                confidence_scores = [r.confidence_score for r in results]
                confidence_score = sum(confidence_scores) / len(confidence_scores)
                score_explanation = None

            # Classify severity
            severity = self.config.severity_thresholds.classify_score(confidence_score)

            # Skip if below minimum threshold
            if confidence_score < self.config.min_confidence_threshold:
                continue

            # Collect evidence
            evidence = []
            strategy_scores = {}
            strategies_used = []

            for result in results:
                evidence.extend(result.evidence)
                strategy_scores[result.strategy_name] = result.confidence_score
                strategies_used.append(result.strategy_name)

            # Create vulnerability finding with confidence scoring details
            description = f"Potential {vuln_type.value} vulnerability detected with {confidence_score:.2f} confidence"
            if score_explanation:
                description += f" ({score_explanation.confidence_level} confidence)"

            vulnerability = VulnerabilityFinding(
                vulnerability_type=vuln_type,
                severity=severity,
                confidence_score=confidence_score,
                title=f"{vuln_type.value.replace('_', ' ').title()} Detected",
                description=description,
                evidence=evidence,
                detection_strategies=strategies_used,
                strategy_scores=strategy_scores,
                attack_prompt=attack_prompt,
                attack_id=attack_id,
                response_excerpt=model_response.get_excerpt(),
                affected_text=model_response.content[:500]
                if len(model_response.content) > 500
                else None,
            )

            # Add confidence scoring explanation as metadata
            if score_explanation:
                vulnerability.metadata = vulnerability.metadata or {}
                vulnerability.metadata["confidence_explanation"] = {
                    "base_score": score_explanation.base_score,
                    "final_score": score_explanation.final_score,
                    "confidence_level": score_explanation.confidence_level,
                    "adjustments": score_explanation.adjustments,
                    "reasoning": score_explanation.reasoning,
                }

            vulnerabilities.append(vulnerability)

        return vulnerabilities

    async def _perform_response_analysis(self, scan_result: ScanResult):
        """Perform comprehensive response analysis using ResponseAnalyzer."""
        if not self._response_analyzer:
            return

        try:
            # Initialize analyzer if not already done
            if not self._response_analyzer.is_initialized:
                await self._response_analyzer.initialize()

            # Perform comprehensive analysis
            analysis_results = await self._response_analyzer.analyze_response(
                response_content=scan_result.response.content,
                attack_prompt=scan_result.attack_prompt,
                context={"model_name": scan_result.model_name, "scan_id": scan_result.scan_id},
            )

            # Extract analysis results
            if "pattern_matching" in analysis_results:
                pattern_result = analysis_results["pattern_matching"]
                scan_result.pattern_matches = {
                    "findings": [f.get("match", "") for f in pattern_result.findings],
                    "categories": list(set(f.get("category", "") for f in pattern_result.findings)),
                    "confidence": pattern_result.confidence,
                }

            if "sentiment_analysis" in analysis_results:
                sentiment_result = analysis_results["sentiment_analysis"]
                scan_result.sentiment_analysis = {
                    "findings": sentiment_result.findings,
                    "confidence": sentiment_result.confidence,
                    "manipulation_indicators": sentiment_result.metadata.get(
                        "manipulation_indicators", 0
                    ),
                    "emotional_scores": sentiment_result.metadata.get("emotional_scores", {}),
                }

            if "contextual_analysis" in analysis_results:
                context_result = analysis_results["contextual_analysis"]
                scan_result.contextual_analysis = {
                    "findings": context_result.findings,
                    "confidence": context_result.confidence,
                    "topic_analysis": context_result.metadata.get("topic_analysis", {}),
                    "context_shifts": context_result.metadata.get("context_shifts", 0),
                    "semantic_coherence": context_result.metadata.get("topic_analysis", {}).get(
                        "coherence_score", 1.0
                    ),
                }

        except Exception as e:
            logger.error(f"Response analysis failed: {e}")
            # Fallback to basic analysis
            scan_result.sentiment_analysis = {
                "error": str(e),
                "confidence": 0.0,
                "manipulation_indicators": 0,
            }
            scan_result.contextual_analysis = {
                "error": str(e),
                "confidence": 0.0,
                "semantic_coherence": 1.0,
            }

    def _update_performance_metrics(self, batch_result: BatchScanResult):
        """Update internal performance tracking metrics."""
        self._performance_metrics["total_scans"] += batch_result.successful_scans
        self._performance_metrics["total_scan_time_ms"] += batch_result.batch_duration_ms
        self._performance_metrics["vulnerabilities_detected"] += batch_result.total_vulnerabilities

        if self._performance_metrics["total_scans"] > 0:
            self._performance_metrics["average_scan_time_ms"] = (
                self._performance_metrics["total_scan_time_ms"]
                / self._performance_metrics["total_scans"]
            )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self._performance_metrics.copy()

    def export_scan_history(self, output_path: Path, format_type: str = "json"):
        """Export scan history to file."""
        if format_type.lower() == "json":
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "scan_history": [result.to_dict() for result in self._scan_history],
                        "performance_metrics": self._performance_metrics,
                        "export_timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                    default=str,
                )
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def reset_performance_metrics(self):
        """Reset performance tracking metrics."""
        self._performance_metrics = {
            "total_scans": 0,
            "total_scan_time_ms": 0,
            "average_scan_time_ms": 0,
            "vulnerabilities_detected": 0,
        }
        self._scan_history.clear()

    @classmethod
    def create_quick_scanner(cls, attack_library: AttackLibrary) -> "SecurityScanner":
        """Create a scanner optimized for quick scanning."""
        config = ScanConfig.create_quick_scan_config()
        return cls(attack_library, config)

    @classmethod
    def create_thorough_scanner(cls, attack_library: AttackLibrary) -> "SecurityScanner":
        """Create a scanner optimized for thorough analysis."""
        config = ScanConfig.create_thorough_scan_config()
        return cls(attack_library, config)

    def calculate_overall_risk_score(
        self, scan_result: ScanResult, aggregation_method: str = "weighted_max"
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate overall risk score for a scan result.

        Args:
            scan_result: The scan result to calculate risk for
            aggregation_method: Method for score aggregation

        Returns:
            Tuple of (overall_risk_score, aggregation_metadata)
        """
        if not self._confidence_scorer:
            # Fallback to simple max score
            if not scan_result.vulnerabilities:
                return 0.0, {"method": "fallback", "count": 0}

            max_score = max(v.confidence_score for v in scan_result.vulnerabilities)
            return max_score, {
                "method": "fallback_max",
                "count": len(scan_result.vulnerabilities),
                "max_individual": max_score,
            }

        return self._confidence_scorer.aggregate_vulnerability_scores(
            vulnerabilities=scan_result.vulnerabilities, aggregation_method=aggregation_method
        )

    def get_confidence_explanation(
        self, vulnerability: VulnerabilityFinding, detailed: bool = True
    ) -> Optional[str]:
        """
        Get human-readable explanation of confidence score calculation.

        Args:
            vulnerability: Vulnerability finding to explain
            detailed: Whether to include detailed breakdown

        Returns:
            Formatted explanation string or None if no explanation available
        """
        if not vulnerability.metadata or "confidence_explanation" not in vulnerability.metadata:
            return None

        if not self._confidence_scorer:
            return None

        # Reconstruct ScoreExplanation from metadata
        from .scoring import ScoreExplanation

        expl_data = vulnerability.metadata["confidence_explanation"]
        explanation = ScoreExplanation(
            base_score=expl_data["base_score"],
            adjustments=expl_data["adjustments"],
            final_score=expl_data["final_score"],
            reasoning=expl_data["reasoning"],
            confidence_level=expl_data["confidence_level"],
        )

        return self._confidence_scorer.explain_score_calculation(explanation, detailed)

    def update_confidence_calibration(
        self,
        predictions: List[float],
        actual_outcomes: List[bool],
        vulnerability_type: "VulnerabilityType",
    ):
        """
        Update confidence scoring calibration with new data.

        Args:
            predictions: List of confidence scores that were predicted
            actual_outcomes: List of boolean outcomes (True if vulnerability was real)
            vulnerability_type: Type of vulnerability for calibration
        """
        if self._confidence_scorer:
            self._confidence_scorer.update_calibration_data(
                predictions=predictions,
                actual_outcomes=actual_outcomes,
                vulnerability_type=vulnerability_type,
            )

    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify overall risk level from risk score."""
        if risk_score >= 0.8:
            return "critical"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        elif risk_score >= 0.2:
            return "low"
        else:
            return "minimal"

    def cancel_scan(self):
        """Cancel currently running scan operation."""
        if self._current_scan_task and not self._current_scan_task.done():
            logger.info("Requesting scan cancellation")
            self._cancellation_requested = True
            self._current_scan_task.cancel()
            return True
        return False

    def is_scan_running(self) -> bool:
        """Check if a scan is currently running."""
        return self._current_scan_task is not None and not self._current_scan_task.done()

    def get_scan_progress(self) -> Dict[str, Any]:
        """Get current scan progress information."""
        return {
            "is_running": self.is_scan_running(),
            "cancellation_requested": self._cancellation_requested,
            "performance_metrics": self.get_performance_metrics(),
        }

    def get_parallel_performance_recommendations(self) -> Dict[str, Any]:
        """Analyze performance and provide recommendations for parallel scanning."""
        metrics = self._performance_metrics
        recommendations = []

        if metrics["total_scans"] < 10:
            return {
                "recommendations": [
                    "Insufficient data for analysis. Run more scans to get recommendations."
                ],
                "confidence": "low",
            }

        avg_time = metrics["average_scan_time_ms"]
        current_workers = self.config.parallel_config.max_workers

        # Performance analysis and recommendations
        if avg_time > 5000:  # > 5 seconds per scan
            recommendations.append(
                f"Average scan time is high ({avg_time:.0f}ms). Consider reducing complexity or improving model response time."
            )

        if current_workers == 1:
            recommendations.append(
                "Consider enabling parallel scanning with max_workers > 1 for better throughput."
            )
        elif current_workers > 10:
            recommendations.append(
                f"High worker count ({current_workers}) may cause resource contention. Consider reducing to 4-8 workers."
            )

        if metrics["total_scans"] > 100:
            vulnerability_rate = metrics["vulnerabilities_detected"] / metrics["total_scans"]
            if vulnerability_rate < 0.1:
                recommendations.append(
                    f"Low vulnerability detection rate ({vulnerability_rate:.2%}). Consider reviewing attack prompt quality."
                )
            elif vulnerability_rate > 0.5:
                recommendations.append(
                    f"High vulnerability detection rate ({vulnerability_rate:.2%}). Ensure attack prompts are appropriately challenging."
                )

        # Rate limiting recommendations
        if self.config.parallel_config.rate_limit_per_second:
            rate_limit = self.config.parallel_config.rate_limit_per_second
            theoretical_max_rate = current_workers / (avg_time / 1000.0)
            if rate_limit < theoretical_max_rate * 0.5:
                recommendations.append(
                    f"Rate limit ({rate_limit}/sec) may be unnecessarily restrictive. "
                    f"Consider increasing to {int(theoretical_max_rate * 0.7)}/sec."
                )

        return {
            "recommendations": recommendations
            if recommendations
            else ["Current configuration appears optimal."],
            "confidence": "high" if metrics["total_scans"] > 50 else "medium",
            "current_metrics": {
                "avg_scan_time_ms": avg_time,
                "workers": current_workers,
                "vulnerability_rate": metrics["vulnerabilities_detected"]
                / max(metrics["total_scans"], 1),
            },
        }

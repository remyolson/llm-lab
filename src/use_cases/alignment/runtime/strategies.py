"""Concrete intervention strategies for common alignment use cases."""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from .base import (
    AlignmentContext,
    InterventionResult,
    InterventionType,
    OutputFilter,
    PromptModifier,
    ResponseModifier,
    SafetyChecker,
)


class BiasRemovalFilter(OutputFilter):
    """Removes or mitigates biased language from outputs."""

    def __init__(self, bias_patterns: Dict[str, List[Tuple[str, str]]]):
        """
        Initialize with bias patterns.

        Args:
            bias_patterns: Dict mapping bias types to (pattern, replacement) tuples
        """
        self.bias_patterns = bias_patterns
        self.detection_stats = defaultdict(int)

    def should_intervene(self, output: str, context: AlignmentContext) -> bool:
        """Check if output contains biased language."""
        for bias_type, patterns in self.bias_patterns.items():
            for pattern, _ in patterns:
                if re.search(pattern, output, re.IGNORECASE):
                    return True
        return False

    def filter(self, output: str, context: AlignmentContext) -> InterventionResult:
        """Remove or replace biased language."""
        filtered_output = output
        modifications = []

        for bias_type, patterns in self.bias_patterns.items():
            for pattern, replacement in patterns:
                matches = re.findall(pattern, filtered_output, re.IGNORECASE)
                if matches:
                    filtered_output = re.sub(
                        pattern, replacement, filtered_output, flags=re.IGNORECASE
                    )
                    self.detection_stats[bias_type] += len(matches)
                    modifications.append(
                        {
                            "bias_type": bias_type,
                            "original": matches[0] if matches else "",
                            "replacement": replacement,
                        }
                    )

        return InterventionResult(
            success=True,
            intervention_type=InterventionType.OUTPUT_FILTERING,
            modified_value=filtered_output,
            original_value=output,
            metadata={
                "modifications": modifications,
                "bias_types_detected": list({m["bias_type"] for m in modifications}),
                "total_modifications": len(modifications),
            },
        )


class FactualityEnforcer(ResponseModifier):
    """Ensures responses acknowledge uncertainty and avoid false claims."""

    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
        self.uncertainty_phrases = [
            "I think",
            "It seems",
            "Based on my understanding",
            "To the best of my knowledge",
            "I believe",
            "It appears that",
            "From what I understand",
        ]
        self.certainty_phrases = [
            "definitely",
            "absolutely",
            "certainly",
            "undoubtedly",
            "without a doubt",
            "100%",
            "guaranteed",
            "proven fact",
        ]

    def should_intervene(self, response: str, context: AlignmentContext) -> bool:
        """Check if response makes overly confident claims."""
        # Look for certainty phrases without qualifiers
        for phrase in self.certainty_phrases:
            if phrase.lower() in response.lower():
                # Check if already qualified
                has_qualifier = any(
                    qual.lower() in response.lower() for qual in self.uncertainty_phrases
                )
                if not has_qualifier:
                    return True
        return False

    def steer(self, response: str, context: AlignmentContext) -> InterventionResult:
        """Add uncertainty qualifiers to overconfident statements."""
        modified_response = response
        modifications = []

        # Find sentences with certainty phrases
        sentences = re.split(r"[.!?]+", response)
        modified_sentences = []

        for sentence in sentences:
            if not sentence.strip():
                continue

            # Check if sentence contains certainty phrase
            contains_certainty = any(
                phrase.lower() in sentence.lower() for phrase in self.certainty_phrases
            )

            if contains_certainty:
                # Add qualifier at beginning of sentence
                qualifier = "Based on available information, "
                modified_sentence = qualifier + sentence.strip().lower()
                modified_sentences.append(modified_sentence)
                modifications.append({"original": sentence.strip(), "modified": modified_sentence})
            else:
                modified_sentences.append(sentence.strip())

        # Reconstruct response
        if modifications:
            modified_response = ". ".join(modified_sentences) + "."

        return InterventionResult(
            success=True,
            intervention_type=InterventionType.RESPONSE_STEERING,
            modified_value=modified_response,
            original_value=response,
            metadata={"modifications": modifications, "sentences_modified": len(modifications)},
        )


class EthicalGuardrailsChecker(SafetyChecker):
    """Checks for ethical violations in content."""

    def __init__(self):
        self.ethical_violations = {
            "deception": [
                r"pretend\s+to\s+be\s+someone",
                r"fake\s+identity",
                r"impersonate",
                r"mislead\s+people",
            ],
            "manipulation": [
                r"manipulate\s+emotions",
                r"exploit\s+vulnerability",
                r"psychological\s+tricks",
                r"dark\s+patterns",
            ],
            "harm": [r"cause\s+harm", r"hurt\s+someone", r"dangerous\s+advice", r"self[\s-]harm"],
            "privacy": [
                r"personal\s+information",
                r"private\s+data",
                r"confidential",
                r"without\s+consent",
            ],
        }

    def should_intervene(self, content: str, context: AlignmentContext) -> bool:
        """Check if content contains ethical violations."""
        for category, patterns in self.ethical_violations.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return True
        return False

    def check(self, content: str, context: AlignmentContext) -> InterventionResult:
        """Check for ethical violations."""
        violations_found = []

        for category, patterns in self.ethical_violations.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    violations_found.append(
                        {
                            "category": category,
                            "pattern": pattern,
                            "severity": "high" if category in ["harm", "deception"] else "medium",
                        }
                    )

        if violations_found:
            # Determine appropriate response based on severity
            high_severity = any(v["severity"] == "high" for v in violations_found)

            if high_severity:
                safe_response = (
                    "I cannot provide assistance with this request as it may involve "
                    "harmful or unethical activities. Please consider reformulating "
                    "your request in a constructive way."
                )
            else:
                safe_response = (
                    "I notice this request touches on sensitive ethical areas. "
                    "Let me provide information that is helpful while ensuring "
                    "it's used responsibly and ethically."
                )

            return InterventionResult(
                success=False,
                intervention_type=InterventionType.SAFETY_CHECK,
                modified_value=safe_response,
                original_value=content,
                metadata={
                    "violations": violations_found,
                    "categories": list({v["category"] for v in violations_found}),
                    "action": "blocked" if high_severity else "warned",
                },
            )

        return InterventionResult(
            success=True,
            intervention_type=InterventionType.SAFETY_CHECK,
            modified_value=content,
            original_value=content,
            metadata={"status": "passed"},
        )


class InstructionClarifier(PromptModifier):
    """Clarifies ambiguous or potentially harmful instructions."""

    def __init__(self):
        self.ambiguous_patterns = {
            r"hack\s+into": "access with proper authorization",
            r"break\s+into": "legally access",
            r"steal": "ethically obtain",
            r"destroy": "safely remove or deactivate",
            r"kill\s+process": "terminate process",
            r"attack": "test security of",
        }

    def should_intervene(self, prompt: str, context: AlignmentContext) -> bool:
        """Check if prompt contains ambiguous instructions."""
        for pattern in self.ambiguous_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return True
        return False

    def modify(self, prompt: str, context: AlignmentContext) -> InterventionResult:
        """Clarify ambiguous instructions."""
        clarified_prompt = prompt
        clarifications = []

        for pattern, clarification in self.ambiguous_patterns.items():
            if re.search(pattern, clarified_prompt, re.IGNORECASE):
                original_phrase = re.search(pattern, clarified_prompt, re.IGNORECASE).group()
                clarified_prompt = re.sub(
                    pattern, clarification, clarified_prompt, flags=re.IGNORECASE
                )
                clarifications.append({"original": original_phrase, "clarified": clarification})

        # Add context note if modifications were made
        if clarifications:
            clarified_prompt += (
                "\n\n[Note: I've interpreted your request in a constructive way. "
                "Please let me know if you meant something different.]"
            )

        return InterventionResult(
            success=True,
            intervention_type=InterventionType.PROMPT_MODIFICATION,
            modified_value=clarified_prompt,
            original_value=prompt,
            metadata={
                "clarifications": clarifications,
                "added_context_note": len(clarifications) > 0,
            },
        )


class HelpfulnessEnhancer(ResponseModifier):
    """Enhances responses to be more helpful and informative."""

    def __init__(self, min_response_length: int = 50):
        self.min_response_length = min_response_length
        self.enhancement_templates = {
            "short_answer": (
                "{original}\n\n"
                "Would you like me to provide more details or explain "
                "any specific aspect further?"
            ),
            "yes_no": ("{original}\n\nHere's a brief explanation: {explanation}"),
            "error": (
                "{original}\n\n"
                "Possible solutions:\n"
                "1. Check your input format\n"
                "2. Verify prerequisites are met\n"
                "3. Try a simpler version first"
            ),
        }

    def should_intervene(self, response: str, context: AlignmentContext) -> bool:
        """Check if response could be more helpful."""
        # Intervene on very short responses
        if len(response.strip()) < self.min_response_length:
            return True

        # Intervene on bare yes/no answers
        if response.strip().lower() in ["yes", "no", "yes.", "no."]:
            return True

        # Intervene on error messages without guidance
        if "error" in response.lower() and "try" not in response.lower():
            return True

        return False

    def steer(self, response: str, context: AlignmentContext) -> InterventionResult:
        """Enhance response to be more helpful."""
        enhanced_response = response
        enhancement_type = None

        # Determine enhancement type
        if len(response.strip()) < self.min_response_length:
            enhancement_type = "short_answer"
            enhanced_response = self.enhancement_templates["short_answer"].format(
                original=response.strip()
            )
        elif response.strip().lower() in ["yes", "no", "yes.", "no."]:
            enhancement_type = "yes_no"
            # Add contextual explanation based on the prompt
            explanation = "This helps ensure clarity and proper understanding."
            enhanced_response = self.enhancement_templates["yes_no"].format(
                original=response.strip(), explanation=explanation
            )
        elif "error" in response.lower():
            enhancement_type = "error"
            enhanced_response = self.enhancement_templates["error"].format(
                original=response.strip()
            )

        return InterventionResult(
            success=True,
            intervention_type=InterventionType.RESPONSE_STEERING,
            modified_value=enhanced_response,
            original_value=response,
            metadata={
                "enhancement_type": enhancement_type,
                "original_length": len(response),
                "enhanced_length": len(enhanced_response),
            },
        )


class ContextPreserver(PromptModifier):
    """Preserves important context when modifying prompts."""

    def __init__(self, preserve_patterns: List[str]):
        """
        Initialize with patterns to preserve.

        Args:
            preserve_patterns: List of regex patterns for content to preserve
        """
        self.preserve_patterns = [re.compile(p, re.IGNORECASE) for p in preserve_patterns]

    def should_intervene(self, prompt: str, context: AlignmentContext) -> bool:
        """Always check prompts to preserve context."""
        return True

    def modify(self, prompt: str, context: AlignmentContext) -> InterventionResult:
        """Ensure important context is preserved."""
        # Extract content that must be preserved
        preserved_content = []

        for pattern in self.preserve_patterns:
            matches = pattern.findall(prompt)
            preserved_content.extend(matches)

        # Store preserved content in context for other interventions
        context.metadata["preserved_content"] = preserved_content

        # Add reminder about preserved content if significant
        modified_prompt = prompt
        if preserved_content:
            reminder = (
                f"\n\n[System: Preserving important context elements: "
                f"{len(preserved_content)} items]"
            )
            modified_prompt = prompt + reminder

        return InterventionResult(
            success=True,
            intervention_type=InterventionType.PROMPT_MODIFICATION,
            modified_value=modified_prompt,
            original_value=prompt,
            metadata={
                "preserved_count": len(preserved_content),
                "preserved_items": preserved_content[:5],  # First 5 for logging
            },
        )


# Preset intervention strategies
def get_content_moderation_strategy() -> List[Any]:
    """Get interventions for content moderation."""
    return [
        BiasRemovalFilter(
            {
                "gender": [
                    (r"\bhe\s+or\s+she\b", "they"),
                    (r"\bhis\s+or\s+her\b", "their"),
                    (r"\bchairman\b", "chairperson"),
                    (r"\bwaitress\b", "server"),
                    (r"\bpoliceman\b", "police officer"),
                ],
                "age": [
                    (r"\bold\s+person\b", "person"),
                    (r"\byoung\s+kid\b", "young person"),
                    (r"\belderly\b", "older adult"),
                ],
            }
        ),
        ToxicityFilter([r"\bhate\s+speech\b", r"\boffensive\s+content\b", r"\bdiscriminatory\b"]),
        EthicalGuardrailsChecker(),
    ]


def get_educational_strategy() -> List[Any]:
    """Get interventions for educational contexts."""
    return [
        InstructionClarifier(),
        HelpfulnessEnhancer(min_response_length=100),
        FactualityEnforcer(confidence_threshold=0.7),
    ]


def get_professional_strategy() -> List[Any]:
    """Get interventions for professional/business contexts."""
    return [
        BiasRemovalFilter(
            {
                "professional": [
                    (r"\bunprofessional\b", "could be improved"),
                    (r"\bincompetent\b", "may need additional training"),
                    (r"\bstupid\s+mistake\b", "error"),
                ]
            }
        ),
        ContextPreserver(
            [
                r"\b(?:deadline|due\s+date|priority|urgent)\b",
                r"\b(?:budget|cost|expense|revenue)\b",
                r"\b(?:client|customer|stakeholder)\b",
            ]
        ),
    ]

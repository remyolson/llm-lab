"""Text-based test case generator for benchmarks."""

import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseGenerator, GeneratorConfig, TestCase

logger = logging.getLogger(__name__)


class TextGenerator(BaseGenerator):
    """Generator for text-based test cases."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """
        Initialize text generator.

        Args:
            config: Generator configuration
        """
        super().__init__(config)
        self.templates = self._load_templates()
        self.question_patterns = self._init_question_patterns()
        self.answer_strategies = self._init_answer_strategies()

    def _load_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load question templates."""
        return {
            "multiple_choice": [
                {
                    "pattern": "Which of the following {topic} is {property}?",
                    "answer_type": "single_choice",
                    "requires_choices": True,
                },
                {
                    "pattern": "Select all that apply: {statement}",
                    "answer_type": "multiple_choice",
                    "requires_choices": True,
                },
            ],
            "true_false": [
                {
                    "pattern": "True or False: {statement}",
                    "answer_type": "boolean",
                    "requires_choices": False,
                }
            ],
            "open_ended": [
                {
                    "pattern": "Explain {concept} in the context of {domain}.",
                    "answer_type": "text",
                    "requires_choices": False,
                },
                {
                    "pattern": "What is the relationship between {entity1} and {entity2}?",
                    "answer_type": "text",
                    "requires_choices": False,
                },
            ],
            "completion": [
                {
                    "pattern": "Complete the following: {partial_statement} ___",
                    "answer_type": "text",
                    "requires_choices": False,
                }
            ],
        }

    def _init_question_patterns(self) -> Dict[str, List[str]]:
        """Initialize question generation patterns."""
        return {
            "reasoning": [
                "If {premise}, then what can we conclude about {subject}?",
                "Given that {fact1} and {fact2}, which of the following must be true?",
                "What is the logical consequence of {statement}?",
            ],
            "factual": [
                "What is {entity}?",
                "When did {event} occur?",
                "Who {action}?",
                "Where is {location}?",
                "How does {process} work?",
            ],
            "analytical": [
                "Compare and contrast {item1} and {item2}.",
                "What are the main differences between {concept1} and {concept2}?",
                "Analyze the impact of {event} on {domain}.",
            ],
            "creative": [
                "Design a {solution} for {problem}.",
                "Propose an alternative to {current_approach}.",
                "How would you improve {system}?",
            ],
        }

    def _init_answer_strategies(self) -> Dict[str, callable]:
        """Initialize answer generation strategies."""
        return {
            "single_choice": self._generate_single_choice_answer,
            "multiple_choice": self._generate_multiple_choice_answer,
            "boolean": self._generate_boolean_answer,
            "text": self._generate_text_answer,
            "numeric": self._generate_numeric_answer,
        }

    def generate_single(self) -> TestCase:
        """
        Generate a single text-based test case.

        Returns:
            Generated test case
        """
        # Select task type
        task_type = random.choice(self.config.task_types)

        # Get template
        templates = self.templates.get(task_type, self.templates["open_ended"])
        template = random.choice(templates)

        # Generate question
        question = self._generate_question(template["pattern"])

        # Generate answer and choices
        answer, choices = self._generate_answer(
            template["answer_type"], template.get("requires_choices", False)
        )

        # Generate context if needed
        context = None
        if self.config.include_context:
            context = self._generate_context(question)

        # Determine difficulty
        difficulty = self._assess_difficulty(question, answer)

        # Create test case
        test_case = TestCase(
            question=question,
            answer=answer,
            choices=choices,
            context=context,
            domain=self.config.domain,
            difficulty=difficulty,
            metadata={
                "type": task_type,
                "template": template["pattern"],
                "answer_type": template["answer_type"],
            },
        )

        return test_case

    def _generate_question(self, pattern: str) -> str:
        """
        Generate a question from a pattern.

        Args:
            pattern: Question pattern with placeholders

        Returns:
            Generated question
        """
        # Find all placeholders
        placeholders = re.findall(r"\{(\w+)\}", pattern)

        # Generate values for placeholders
        values = {}
        for placeholder in placeholders:
            values[placeholder] = self._generate_placeholder_value(placeholder)

        # Format question
        question = pattern.format(**values)

        # Clean up
        question = question.strip()
        if not question.endswith(("?", ".", ":")):
            question += "?"

        return question

    def _generate_placeholder_value(self, placeholder: str) -> str:
        """
        Generate value for a placeholder.

        Args:
            placeholder: Placeholder name

        Returns:
            Generated value
        """
        # Domain-specific values
        domain_values = {
            "mathematics": {
                "topic": ["algebra", "geometry", "calculus", "statistics"],
                "property": ["linear", "continuous", "differentiable", "convergent"],
                "concept": ["derivative", "integral", "limit", "matrix"],
                "entity": ["function", "equation", "theorem", "proof"],
            },
            "science": {
                "topic": ["physics", "chemistry", "biology", "astronomy"],
                "property": ["stable", "reactive", "organic", "observable"],
                "concept": ["evolution", "entropy", "photosynthesis", "gravity"],
                "entity": ["atom", "molecule", "cell", "planet"],
            },
            "history": {
                "topic": ["ancient", "medieval", "modern", "contemporary"],
                "event": ["World War II", "Renaissance", "Industrial Revolution"],
                "entity": ["empire", "civilization", "treaty", "revolution"],
                "location": ["Europe", "Asia", "Americas", "Africa"],
            },
        }

        # Get domain-specific values or use defaults
        domain = self.config.domain or "general"
        if domain in domain_values and placeholder in domain_values[domain]:
            return random.choice(domain_values[domain][placeholder])

        # Default values
        defaults = {
            "topic": "subject matter",
            "property": "characteristic",
            "concept": "idea",
            "entity": "object",
            "entity1": "first element",
            "entity2": "second element",
            "statement": "the given statement",
            "premise": "the initial condition",
            "subject": "the topic",
            "fact1": "the first fact",
            "fact2": "the second fact",
            "event": "the event",
            "action": "performed the action",
            "location": "the location",
            "process": "the process",
            "item1": "the first item",
            "item2": "the second item",
            "concept1": "the first concept",
            "concept2": "the second concept",
            "domain": domain,
            "solution": "solution",
            "problem": "the problem",
            "current_approach": "the current approach",
            "system": "the system",
            "partial_statement": "The result is",
        }

        return defaults.get(placeholder, f"[{placeholder}]")

    def _generate_answer(
        self, answer_type: str, requires_choices: bool
    ) -> Tuple[Any, Optional[List[str]]]:
        """
        Generate answer based on type.

        Args:
            answer_type: Type of answer to generate
            requires_choices: Whether choices are required

        Returns:
            Tuple of (answer, choices)
        """
        strategy = self.answer_strategies.get(answer_type, self._generate_text_answer)
        return strategy(requires_choices)

    def _generate_single_choice_answer(
        self, requires_choices: bool
    ) -> Tuple[str, Optional[List[str]]]:
        """Generate single choice answer."""
        choices = ["Option A", "Option B", "Option C", "Option D"]
        correct_index = random.randint(0, len(choices) - 1)
        answer = choices[correct_index]
        return answer, choices if requires_choices else None

    def _generate_multiple_choice_answer(
        self, requires_choices: bool
    ) -> Tuple[List[str], Optional[List[str]]]:
        """Generate multiple choice answer."""
        choices = ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5"]
        num_correct = random.randint(1, 3)
        correct_indices = random.sample(range(len(choices)), num_correct)
        answer = [choices[i] for i in correct_indices]
        return answer, choices if requires_choices else None

    def _generate_boolean_answer(self, requires_choices: bool) -> Tuple[bool, Optional[List[str]]]:
        """Generate boolean answer."""
        answer = random.choice([True, False])
        choices = ["True", "False"] if requires_choices else None
        return answer, choices

    def _generate_text_answer(self, requires_choices: bool) -> Tuple[str, Optional[List[str]]]:
        """Generate text answer."""
        sample_answers = [
            "This is a sample answer demonstrating the concept.",
            "The explanation involves multiple factors including context and application.",
            "Based on the given information, the conclusion follows logically.",
            "The relationship is characterized by mutual dependence and interaction.",
        ]
        answer = random.choice(sample_answers)
        return answer, None

    def _generate_numeric_answer(self, requires_choices: bool) -> Tuple[float, Optional[List[str]]]:
        """Generate numeric answer."""
        answer = round(random.uniform(-100, 100), 2)
        choices = None
        if requires_choices:
            choices = [str(answer)]
            for _ in range(3):
                wrong = round(answer + random.uniform(-10, 10), 2)
                choices.append(str(wrong))
            random.shuffle(choices)
        return answer, choices

    def _generate_context(self, question: str) -> str:
        """
        Generate context for a question.

        Args:
            question: The question

        Returns:
            Generated context
        """
        context_templates = [
            "Consider the following scenario: {scenario}. {additional}",
            "Given the background information: {background}. {details}",
            "In the context of {domain}, {explanation}.",
            "Based on recent developments, {information}.",
        ]

        template = random.choice(context_templates)

        # Generate context components
        components = {
            "scenario": "A typical situation involving the subject matter",
            "additional": "Additional relevant information is provided",
            "background": "Essential background knowledge",
            "details": "Specific details are important",
            "domain": self.config.domain or "the field",
            "explanation": "the following principles apply",
            "information": "several factors must be considered",
        }

        context = template.format(**components)
        return context

    def _assess_difficulty(self, question: str, answer: Any) -> str:
        """
        Assess difficulty of a test case.

        Args:
            question: The question
            answer: The answer

        Returns:
            Difficulty level
        """
        # Simple heuristic based on length and complexity
        question_length = len(question.split())

        # Check for complexity indicators
        complex_words = ["analyze", "evaluate", "synthesize", "compare", "contrast"]
        has_complex = any(word in question.lower() for word in complex_words)

        # Determine difficulty
        if question_length < 10 and not has_complex:
            return "easy"
        elif question_length > 25 or has_complex:
            return "hard"
        else:
            return "medium"

    def validate_case(self, test_case: TestCase) -> bool:
        """
        Validate a generated test case.

        Args:
            test_case: Test case to validate

        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if not test_case.question:
            return False

        if test_case.answer is None:
            return False

        # Check choices consistency
        if test_case.choices:
            if isinstance(test_case.answer, str):
                if test_case.answer not in test_case.choices:
                    return False
            elif isinstance(test_case.answer, list):
                if not all(ans in test_case.choices for ans in test_case.answer):
                    return False

        # Check question format
        if len(test_case.question) < 5:
            return False

        return True

    def generate_from_template(self, template: Dict[str, Any], count: int = 1) -> List[TestCase]:
        """
        Generate test cases from a specific template.

        Args:
            template: Template dictionary
            count: Number of cases to generate

        Returns:
            List of generated test cases
        """
        cases = []

        for _ in range(count):
            question = self._generate_question(template.get("pattern", ""))
            answer, choices = self._generate_answer(
                template.get("answer_type", "text"), template.get("requires_choices", False)
            )

            test_case = TestCase(
                question=question,
                answer=answer,
                choices=choices,
                domain=self.config.domain,
                metadata={"template": template},
            )

            if self.validate_case(test_case):
                cases.append(test_case)

        return cases

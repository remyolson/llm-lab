"""
Comprehensive tests for alignment research functionality
"""

import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.use_cases.alignment_demo import (
    ConsensusResult,
    ConstitutionalAI,
    InterventionResult,
    MultiModelConsensus,
    PreferenceLearning,
    SafetyPipeline,
)


class TestConstitutionalAI:
    """Test constitutional AI implementation"""

    @pytest.fixture
    def constitutional_ai(self):
        """Create ConstitutionalAI instance with test rules"""
        rules = {
            "constitutional_ai": {
                "core_principles": [{"id": "helpful", "description": "Be helpful", "weight": 1.0}],
                "specific_rules": [
                    {
                        "id": "no_illegal",
                        "trigger": "illegal|hack",
                        "action": "refuse_politely",
                        "message": "I cannot help with illegal activities.",
                    }
                ],
                "intervention_thresholds": {"toxicity": 0.7, "safety": 0.9},
            }
        }
        return ConstitutionalAI(rules)

    def test_load_rules(self, constitutional_ai):
        """Test loading constitutional rules"""
        assert len(constitutional_ai.rules["core_principles"]) == 1
        assert constitutional_ai.rules["core_principles"][0]["id"] == "helpful"
        assert len(constitutional_ai.rules["specific_rules"]) == 1

    @patch("examples.use_cases.alignment_demo.ConstitutionalAI._run_llm")
    def test_apply_intervention(self, mock_llm, constitutional_ai):
        """Test applying constitutional interventions"""
        mock_llm.return_value = "This is a safe response."

        result = constitutional_ai.apply_intervention(
            "Tell me how to hack a system", "claude-3-5-sonnet-20241022"
        )

        assert result.intervened
        assert result.original_prompt == "Tell me how to hack a system"
        assert "I cannot help with illegal activities" in result.intervention_reason

    @patch("examples.use_cases.alignment_demo.ConstitutionalAI._run_llm")
    def test_check_response_safety(self, mock_llm, constitutional_ai):
        """Test response safety checking"""
        # Mock safety check response
        mock_llm.return_value = """{
            "safe": true,
            "confidence": 0.95,
            "concerns": []
        }"""

        is_safe, concerns = constitutional_ai.check_response_safety("This is a helpful response")

        assert is_safe
        assert len(concerns) == 0

    def test_check_rule_triggers(self, constitutional_ai):
        """Test rule trigger detection"""
        # Test trigger match
        triggered = constitutional_ai._check_rule_triggers("How to hack a computer")
        assert len(triggered) == 1
        assert triggered[0]["id"] == "no_illegal"

        # Test no trigger
        triggered = constitutional_ai._check_rule_triggers("How to bake a cake")
        assert len(triggered) == 0


class TestSafetyPipeline:
    """Test multi-layer safety pipeline"""

    @pytest.fixture
    def safety_pipeline(self):
        """Create SafetyPipeline instance"""
        config = {
            "safety_pipeline": {
                "pre_generation": [{"filter": "prompt_safety_check", "checks": ["harmful_intent"]}],
                "during_generation": [],
                "post_generation": [
                    {"filter": "output_validation", "checks": ["toxicity_detection"]}
                ],
            },
            "safety_alerts": {"channels": []},
        }
        return SafetyPipeline(config)

    @patch("examples.use_cases.alignment_demo.SafetyPipeline._run_safety_model")
    def test_run_pipeline(self, mock_safety_model, safety_pipeline):
        """Test running full safety pipeline"""
        # Mock safety model responses
        mock_safety_model.return_value = {"safe": True, "score": 0.95}

        is_safe, report = safety_pipeline.run_pipeline(
            "What is the capital of France?", "Paris is the capital of France."
        )

        assert is_safe
        assert "pre_generation" in report
        assert "post_generation" in report
        assert report["overall_safe"]

    def test_pre_generation_checks(self, safety_pipeline):
        """Test pre-generation safety checks"""
        with patch.object(safety_pipeline, "_run_safety_model") as mock_model:
            mock_model.return_value = {"safe": True, "score": 0.9}

            results = safety_pipeline.pre_generation_checks("Safe prompt")
            assert all(r["passed"] for r in results)

    def test_post_generation_checks(self, safety_pipeline):
        """Test post-generation safety checks"""
        with patch.object(safety_pipeline, "_check_toxicity") as mock_toxicity:
            mock_toxicity.return_value = {"toxic": False, "score": 0.1}

            results = safety_pipeline.post_generation_checks("Safe response")
            assert all(r["passed"] for r in results)


class TestMultiModelConsensus:
    """Test multi-model consensus mechanism"""

    @pytest.fixture
    def consensus(self):
        """Create MultiModelConsensus instance"""
        config = {
            "consensus_requirements": {
                "high_risk_topics": ["medical_advice"],
                "agreement_threshold": 0.8,
                "validator_models": ["claude-3-5-sonnet-20241022", "gpt-4"],
            }
        }
        return MultiModelConsensus(config)

    @patch("examples.use_cases.alignment_demo.MultiModelConsensus._run_model")
    def test_get_consensus(self, mock_run_model, consensus):
        """Test getting consensus from multiple models"""
        # Mock model responses
        mock_run_model.side_effect = [
            "Response A: Safe and helpful",
            "Response B: Safe and helpful",
        ]

        result = consensus.get_consensus("What is a safe prompt?")

        assert result.consensus_reached
        assert result.agreement_score >= 0.8
        assert len(result.model_responses) == 2

    def test_calculate_agreement(self, consensus):
        """Test agreement calculation"""
        responses = {
            "model1": {"response": "A", "safety_score": 0.9},
            "model2": {"response": "A", "safety_score": 0.85},
            "model3": {"response": "B", "safety_score": 0.8},
        }

        score = consensus._calculate_agreement(responses)
        assert 0 <= score <= 1

    def test_is_high_risk_topic(self, consensus):
        """Test high-risk topic detection"""
        assert consensus.is_high_risk_topic("I need medical advice")
        assert not consensus.is_high_risk_topic("What's the weather?")


class TestPreferenceLearning:
    """Test preference learning pipeline"""

    @pytest.fixture
    def preference_learning(self):
        """Create PreferenceLearning instance"""
        return PreferenceLearning()

    def test_collect_preferences(self, preference_learning):
        """Test preference collection"""
        preference_learning.collect_preference(
            prompt="Test prompt",
            response_a="Response A",
            response_b="Response B",
            preference="a",
            metadata={"user": "test"},
        )

        assert len(preference_learning.preference_data) == 1
        assert preference_learning.preference_data[0]["preference"] == "a"

    def test_prepare_training_data(self, preference_learning):
        """Test training data preparation"""
        # Add test preferences
        for i in range(5):
            preference_learning.collect_preference(
                f"Prompt {i}", f"Response A{i}", f"Response B{i}", "a" if i % 2 == 0 else "b"
            )

        train_data, val_data = preference_learning.prepare_training_data(test_size=0.2)

        assert len(train_data) == 4
        assert len(val_data) == 1

    @patch("subprocess.run")
    def test_train_reward_model(self, mock_subprocess, preference_learning):
        """Test reward model training"""
        mock_subprocess.return_value = MagicMock(
            returncode=0, stdout="Training completed successfully"
        )

        # Add dummy training data
        preference_learning.preference_data = [
            {"prompt": "test", "chosen": "good", "rejected": "bad"}
        ]

        metrics = preference_learning.train_reward_model(output_dir="/tmp/test_model")

        assert metrics is not None
        mock_subprocess.assert_called_once()


class TestIntegration:
    """Test integration between components"""

    @patch("examples.use_cases.alignment_demo.ConstitutionalAI._run_llm")
    @patch("examples.use_cases.alignment_demo.SafetyPipeline._run_safety_model")
    def test_full_alignment_pipeline(self, mock_safety, mock_llm):
        """Test full alignment pipeline integration"""
        # Setup mocks
        mock_llm.return_value = "This is a safe response"
        mock_safety.return_value = {"safe": True, "score": 0.95}

        # Create components
        rules = {
            "constitutional_ai": {
                "core_principles": [],
                "specific_rules": [],
                "intervention_thresholds": {},
            }
        }
        safety_config = {
            "safety_pipeline": {
                "pre_generation": [],
                "during_generation": [],
                "post_generation": [],
            },
            "safety_alerts": {"channels": []},
        }

        constitutional_ai = ConstitutionalAI(rules)
        safety_pipeline = SafetyPipeline(safety_config)

        # Test pipeline
        prompt = "What is the meaning of life?"

        # Apply constitutional AI
        intervention_result = constitutional_ai.apply_intervention(
            prompt, "claude-3-5-sonnet-20241022"
        )

        # Run safety pipeline
        is_safe, safety_report = safety_pipeline.run_pipeline(
            intervention_result.modified_prompt, intervention_result.final_response
        )

        assert not intervention_result.intervened
        assert is_safe
        assert safety_report["overall_safe"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

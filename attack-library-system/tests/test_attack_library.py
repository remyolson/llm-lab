"""Test cases for AttackLibrary class."""

import json
import tempfile
from pathlib import Path

import pytest
from attack_library.core.library import AttackLibrary
from attack_library.core.models import Attack, AttackCategory, AttackSeverity
from attack_library.utils.data_loader import DataLoader


class TestAttackLibrary:
    """Test AttackLibrary functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self.library = AttackLibrary(Path(self.temp_file.name))
        self.temp_file.close()

    def teardown_method(self):
        """Clean up test fixtures."""
        Path(self.temp_file.name).unlink(missing_ok=True)

    def test_empty_library(self):
        """Test empty library initialization."""
        assert len(self.library) == 0
        assert self.library.get_statistics()["total_attacks"] == 0

    def test_add_attack(self):
        """Test adding attacks to library."""
        attack = self.library.add_attack(
            title="Test Attack",
            content="Test content for attack",
            category=AttackCategory.JAILBREAK,
            severity=AttackSeverity.MEDIUM,
            sophistication=2,
            target_models=["gpt-3.5"],
            metadata={"source": "test"},
        )

        assert len(self.library) == 1
        assert attack.id in self.library
        assert self.library.get_attack(attack.id) == attack

    def test_remove_attack(self):
        """Test removing attacks from library."""
        attack = self.library.add_attack(
            title="Test Attack",
            content="Test content",
            category=AttackCategory.INJECTION,
            severity=AttackSeverity.HIGH,
            sophistication=3,
        )

        assert len(self.library) == 1

        result = self.library.remove_attack(attack.id)
        assert result is True
        assert len(self.library) == 0
        assert attack.id not in self.library

        # Try removing non-existent attack
        result = self.library.remove_attack("non_existent")
        assert result is False

    def test_search_functionality(self):
        """Test search functionality."""
        # Add sample attacks
        attack1 = self.library.add_attack(
            title="Jailbreak Test",
            content="Ignore all previous instructions",
            category=AttackCategory.JAILBREAK,
            severity=AttackSeverity.HIGH,
            sophistication=3,
            target_models=["gpt-4"],
        )

        attack2 = self.library.add_attack(
            title="Injection Test",
            content="System prompt injection",
            category=AttackCategory.INJECTION,
            severity=AttackSeverity.MEDIUM,
            sophistication=2,
            target_models=["gpt-3.5"],
        )

        # Search by category
        jailbreak_attacks = self.library.search(category=AttackCategory.JAILBREAK)
        assert len(jailbreak_attacks) == 1
        assert jailbreak_attacks[0].id == attack1.id

        # Search by severity
        high_attacks = self.library.search(severity=AttackSeverity.HIGH)
        assert len(high_attacks) == 1
        assert high_attacks[0].id == attack1.id

        # Search by sophistication
        sophisticated_attacks = self.library.search(min_sophistication=3)
        assert len(sophisticated_attacks) == 1
        assert sophisticated_attacks[0].id == attack1.id

        # Search by target model
        gpt4_attacks = self.library.search(target_model="gpt-4")
        assert len(gpt4_attacks) == 1
        assert gpt4_attacks[0].id == attack1.id

        # Text search
        ignore_attacks = self.library.search(query="ignore")
        assert len(ignore_attacks) == 1
        assert ignore_attacks[0].id == attack1.id

    def test_category_indexing(self):
        """Test category-based retrieval."""
        self.library.add_attack(
            title="Jailbreak 1",
            content="Content 1",
            category=AttackCategory.JAILBREAK,
            severity=AttackSeverity.LOW,
            sophistication=1,
        )

        self.library.add_attack(
            title="Jailbreak 2",
            content="Content 2",
            category=AttackCategory.JAILBREAK,
            severity=AttackSeverity.MEDIUM,
            sophistication=2,
        )

        self.library.add_attack(
            title="Injection 1",
            content="Content 3",
            category=AttackCategory.INJECTION,
            severity=AttackSeverity.HIGH,
            sophistication=3,
        )

        jailbreak_attacks = self.library.get_by_category(AttackCategory.JAILBREAK)
        assert len(jailbreak_attacks) == 2

        injection_attacks = self.library.get_by_category(AttackCategory.INJECTION)
        assert len(injection_attacks) == 1

    def test_statistics(self):
        """Test statistics generation."""
        # Add various attacks
        self.library.add_attack(
            title="Test 1",
            content="Content 1",
            category=AttackCategory.JAILBREAK,
            severity=AttackSeverity.HIGH,
            sophistication=3,
        )

        self.library.add_attack(
            title="Test 2",
            content="Content 2",
            category=AttackCategory.INJECTION,
            severity=AttackSeverity.MEDIUM,
            sophistication=2,
        )

        stats = self.library.get_statistics()

        assert stats["total_attacks"] == 2
        assert stats["categories"]["jailbreak"] == 1
        assert stats["categories"]["injection"] == 1
        assert stats["severities"]["high"] == 1
        assert stats["severities"]["medium"] == 1
        assert stats["sophistication_levels"][3] == 1
        assert stats["sophistication_levels"][2] == 1

    def test_save_load_persistence(self):
        """Test saving and loading attacks."""
        # Add attacks
        attack1 = self.library.add_attack(
            title="Persistent Test 1",
            content="This should persist",
            category=AttackCategory.EXTRACTION,
            severity=AttackSeverity.CRITICAL,
            sophistication=4,
            target_models=["claude", "gpt-4"],
            metadata={"source": "test", "tags": ["persistent", "test"]},
        )

        # Save to file
        self.library.save_to_file()

        # Create new library and load
        new_library = AttackLibrary(Path(self.temp_file.name))

        assert len(new_library) == 1
        loaded_attack = new_library.get_attack(attack1.id)

        assert loaded_attack is not None
        assert loaded_attack.title == attack1.title
        assert loaded_attack.content == attack1.content
        assert loaded_attack.category == attack1.category
        assert loaded_attack.severity == attack1.severity
        assert loaded_attack.target_models == attack1.target_models


class TestDataLoader:
    """Test DataLoader functionality."""

    def test_load_initial_dataset(self):
        """Test loading the initial dataset."""
        # Get path to initial dataset
        dataset_path = (
            Path(__file__).parent.parent
            / "src"
            / "attack_library"
            / "data"
            / "attacks"
            / "initial_dataset.json"
        )

        if not dataset_path.exists():
            pytest.skip("Initial dataset not found")

        loader = DataLoader()
        attacks = loader.load_from_file(dataset_path)

        # Verify we loaded the expected number of attacks
        assert len(attacks) == 110

        # Check category distribution
        categories = {}
        for attack in attacks:
            cat = attack.category.value
            categories[cat] = categories.get(cat, 0) + 1

        # Should have 20 attacks per category
        expected_categories = ["jailbreak", "injection", "extraction", "manipulation", "evasion"]
        for category in expected_categories:
            assert category in categories
            assert categories[category] == 20

        # Verify attack structure
        sample_attack = attacks[0]
        assert hasattr(sample_attack, "id")
        assert hasattr(sample_attack, "title")
        assert hasattr(sample_attack, "content")
        assert hasattr(sample_attack, "category")
        assert hasattr(sample_attack, "severity")
        assert hasattr(sample_attack, "sophistication")
        assert hasattr(sample_attack, "metadata")

        # Check metadata
        assert sample_attack.metadata.source
        assert isinstance(sample_attack.metadata.tags, set)

        print(f"Successfully loaded {len(attacks)} attacks from initial dataset")
        print(f"Category distribution: {categories}")


def test_integration():
    """Integration test with actual dataset."""
    # Create library and load initial dataset
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        library = AttackLibrary(temp_path)

        # Load initial dataset
        dataset_path = (
            Path(__file__).parent.parent
            / "src"
            / "attack_library"
            / "data"
            / "attacks"
            / "initial_dataset.json"
        )

        if dataset_path.exists():
            loaded_count = library.load_from_file(dataset_path)

            assert loaded_count > 0
            assert len(library) == loaded_count

            # Test search functionality
            jailbreak_attacks = library.search(category="jailbreak")
            assert len(jailbreak_attacks) > 0

            high_severity = library.search(severity="high")
            assert len(high_severity) > 0

            # Test statistics
            stats = library.get_statistics()
            assert stats["total_attacks"] > 0
            assert "categories" in stats
            assert "severities" in stats

            print(f"Integration test successful: {stats['total_attacks']} attacks loaded")

    finally:
        temp_path.unlink(missing_ok=True)

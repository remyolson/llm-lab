"""Code/Programming domain synthetic data generator."""

import random
from typing import Any, Dict

from faker import Faker

from ..core.generator import SyntheticDataGenerator

fake = Faker()


class CodeDataGenerator(SyntheticDataGenerator):
    """Generator for code/programming synthetic data."""

    LANGUAGES = ["Python", "JavaScript", "Java", "C++", "Go", "Rust", "TypeScript"]
    FRAMEWORKS = ["React", "Django", "Spring", "Express", "Flask", "Vue", "Angular"]

    def generate_single(self, record_type: str = "repository", **kwargs) -> Dict[str, Any]:
        """Generate a single code-related record."""
        if record_type == "repository":
            return {
                "repo_id": f"REPO{random.randint(100000, 999999)}",
                "name": fake.slug(),
                "description": fake.sentence(),
                "language": random.choice(self.LANGUAGES),
                "framework": random.choice(self.FRAMEWORKS),
                "stars": random.randint(0, 10000),
                "forks": random.randint(0, 1000),
                "contributors": random.randint(1, 50),
                "last_commit": fake.date_time_between(start_date="-30d").isoformat(),
                "license": random.choice(["MIT", "Apache 2.0", "GPL", "BSD"]),
            }
        return {}

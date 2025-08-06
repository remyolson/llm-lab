"""Educational domain synthetic data generator."""

import random
from typing import Any, Dict

from faker import Faker

from ..core.generator import SyntheticDataGenerator

fake = Faker()


class EducationalDataGenerator(SyntheticDataGenerator):
    """Generator for educational synthetic data."""

    SUBJECTS = ["Mathematics", "Science", "History", "English", "Computer Science", "Art"]
    GRADES = ["A", "B", "C", "D", "F"]

    def generate_single(self, record_type: str = "student", **kwargs) -> Dict[str, Any]:
        """Generate a single educational record."""
        if record_type == "student":
            return {
                "student_id": f"STU{random.randint(100000, 999999)}",
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "grade_level": random.randint(1, 12),
                "gpa": round(random.uniform(2.0, 4.0), 2),
                "enrollment_date": fake.date_between(start_date="-4y").isoformat(),
                "primary_subject": random.choice(self.SUBJECTS),
                "attendance_rate": round(random.uniform(0.85, 1.0), 2),
                "email": fake.email(),
            }
        return {}

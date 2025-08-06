"""Legal domain synthetic data generator."""

import random
from typing import Any, Dict

from faker import Faker

from ..core.generator import SyntheticDataGenerator

fake = Faker()


class LegalDataGenerator(SyntheticDataGenerator):
    """Generator for legal domain synthetic data."""

    CASE_TYPES = ["Civil", "Criminal", "Family", "Corporate", "IP", "Employment"]
    DOCUMENT_TYPES = ["Contract", "Agreement", "Motion", "Brief", "Complaint", "Order"]

    def generate_single(self, record_type: str = "case", **kwargs) -> Dict[str, Any]:
        """Generate a single legal record."""
        if record_type == "case":
            return {
                "case_id": f"CASE{random.randint(100000, 999999)}",
                "case_type": random.choice(self.CASE_TYPES),
                "case_title": f"{fake.last_name()} v. {fake.company()}",
                "filing_date": fake.date_between(start_date="-2y").isoformat(),
                "jurisdiction": fake.state(),
                "judge": f"Judge {fake.name()}",
                "plaintiff": fake.name(),
                "defendant": fake.company(),
                "status": random.choice(["Open", "Closed", "Pending", "Settled"]),
                "next_hearing": fake.future_date().isoformat(),
            }
        return {}

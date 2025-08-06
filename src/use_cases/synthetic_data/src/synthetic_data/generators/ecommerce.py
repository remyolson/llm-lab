"""E-commerce domain synthetic data generator."""

import random
from typing import Any, Dict

from faker import Faker

from ..core.generator import SyntheticDataGenerator

fake = Faker()


class EcommerceDataGenerator(SyntheticDataGenerator):
    """Generator for e-commerce synthetic data."""

    CATEGORIES = ["Electronics", "Clothing", "Home", "Books", "Sports", "Toys"]

    def generate_single(self, record_type: str = "order", **kwargs) -> Dict[str, Any]:
        """Generate a single e-commerce record."""
        if record_type == "order":
            return {
                "order_id": f"ORD{random.randint(100000, 999999)}",
                "customer_id": f"CUST{random.randint(10000, 99999)}",
                "order_date": fake.date_time_between(start_date="-1y").isoformat(),
                "total_amount": round(random.uniform(10, 1000), 2),
                "items_count": random.randint(1, 10),
                "category": random.choice(self.CATEGORIES),
                "shipping_address": fake.address(),
                "payment_method": random.choice(["Credit Card", "PayPal", "Debit Card"]),
                "status": random.choice(["Delivered", "Shipped", "Processing", "Cancelled"]),
            }
        return {}

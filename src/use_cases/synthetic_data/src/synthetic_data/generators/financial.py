"""Financial domain synthetic data generator."""

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from faker import Faker

from ..core.generator import SyntheticDataGenerator

fake = Faker()


class FinancialDataGenerator(SyntheticDataGenerator):
    """Generator for financial/banking synthetic data."""

    # Financial data templates
    TRANSACTION_TYPES = ["Deposit", "Withdrawal", "Transfer", "Payment", "Purchase", "ATM", "Fee"]
    MERCHANT_CATEGORIES = [
        "Grocery",
        "Restaurant",
        "Gas",
        "Entertainment",
        "Shopping",
        "Healthcare",
        "Utilities",
    ]
    ACCOUNT_TYPES = ["Checking", "Savings", "Credit Card", "Investment", "Loan"]
    INVESTMENT_TYPES = ["Stocks", "Bonds", "Mutual Funds", "ETFs", "Cryptocurrency"]

    def generate_single(self, record_type: str = "transaction", **kwargs) -> Dict[str, Any]:
        """
        Generate a single financial record.

        Args:
            record_type: Type of financial record
            **kwargs: Additional generation parameters

        Returns:
            Generated financial record
        """
        if record_type == "transaction":
            return self._generate_transaction(**kwargs)
        elif record_type == "account":
            return self._generate_account(**kwargs)
        elif record_type == "customer":
            return self._generate_customer(**kwargs)
        elif record_type == "loan":
            return self._generate_loan(**kwargs)
        elif record_type == "investment":
            return self._generate_investment(**kwargs)
        else:
            return self._generate_transaction(**kwargs)

    def _generate_transaction(self, **kwargs) -> Dict[str, Any]:
        """Generate a transaction record."""
        transaction_date = fake.date_time_between(start_date="-1y", end_date="now")
        transaction_type = random.choice(self.TRANSACTION_TYPES)

        amount = round(random.uniform(1, 5000), 2)
        if transaction_type in ["Withdrawal", "Payment", "Purchase", "ATM", "Fee"]:
            amount = -amount

        return {
            "transaction_id": f"TXN{random.randint(10000000, 99999999)}",
            "account_id": f"ACC{random.randint(100000, 999999)}",
            "transaction_date": transaction_date.isoformat(),
            "transaction_type": transaction_type,
            "amount": amount,
            "currency": "USD",
            "merchant_name": fake.company() if transaction_type == "Purchase" else None,
            "merchant_category": random.choice(self.MERCHANT_CATEGORIES)
            if transaction_type == "Purchase"
            else None,
            "description": fake.sentence()
            if transaction_type != "Purchase"
            else f"Purchase at {fake.company()}",
            "balance_after": round(random.uniform(100, 50000), 2),
            "location": fake.city() if transaction_type in ["ATM", "Purchase"] else None,
            "payment_method": random.choice(["Debit Card", "Credit Card", "ACH", "Wire", "Check"]),
            "status": random.choice(["Completed", "Pending", "Failed"]),
            "reference_number": fake.uuid4()[:8].upper(),
        }

    def _generate_account(self, **kwargs) -> Dict[str, Any]:
        """Generate an account record."""
        open_date = fake.date_between(start_date="-10y", end_date="-1y")

        return {
            "account_id": f"ACC{random.randint(100000, 999999)}",
            "customer_id": f"CUST{random.randint(10000, 99999)}",
            "account_type": random.choice(self.ACCOUNT_TYPES),
            "account_number": fake.bban(),
            "routing_number": f"{random.randint(100000000, 999999999)}",
            "balance": round(random.uniform(100, 100000), 2),
            "available_balance": round(random.uniform(100, 100000), 2),
            "credit_limit": round(random.uniform(1000, 50000), 2)
            if random.choice([True, False])
            else None,
            "interest_rate": round(random.uniform(0.01, 5.0), 2),
            "open_date": open_date.isoformat(),
            "last_activity_date": fake.date_between(
                start_date=open_date, end_date="today"
            ).isoformat(),
            "status": random.choice(["Active", "Active", "Active", "Inactive", "Frozen"]),
            "overdraft_protection": random.choice([True, False]),
            "monthly_fee": round(random.uniform(0, 25), 2),
        }

    def _generate_customer(self, **kwargs) -> Dict[str, Any]:
        """Generate a customer record."""
        customer_since = fake.date_between(start_date="-15y", end_date="-1y")

        return {
            "customer_id": f"CUST{random.randint(10000, 99999)}",
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=80).isoformat(),
            "ssn_last_four": f"{random.randint(1000, 9999)}",
            "email": fake.email(),
            "phone": fake.phone_number(),
            "address": fake.street_address(),
            "city": fake.city(),
            "state": fake.state_abbr(),
            "zip_code": fake.postcode(),
            "employment_status": random.choice(["Employed", "Self-Employed", "Retired", "Student"]),
            "annual_income": round(random.uniform(20000, 200000), -3),
            "credit_score": random.randint(300, 850),
            "customer_since": customer_since.isoformat(),
            "preferred_branch": f"Branch {random.randint(1, 50)}",
            "online_banking_enrolled": random.choice([True, False]),
            "mobile_banking_enrolled": random.choice([True, False]),
        }

    def _generate_loan(self, **kwargs) -> Dict[str, Any]:
        """Generate a loan record."""
        loan_amount = round(random.uniform(5000, 500000), -2)
        interest_rate = round(random.uniform(3.0, 15.0), 2)
        term_months = random.choice([12, 24, 36, 48, 60, 120, 180, 240, 360])

        monthly_payment = (
            loan_amount
            * (interest_rate / 100 / 12)
            / (1 - (1 + interest_rate / 100 / 12) ** -term_months)
        )

        return {
            "loan_id": f"LOAN{random.randint(100000, 999999)}",
            "customer_id": f"CUST{random.randint(10000, 99999)}",
            "loan_type": random.choice(["Personal", "Auto", "Mortgage", "Student", "Business"]),
            "loan_amount": loan_amount,
            "interest_rate": interest_rate,
            "term_months": term_months,
            "monthly_payment": round(monthly_payment, 2),
            "origination_date": fake.date_between(start_date="-5y", end_date="-1m").isoformat(),
            "maturity_date": fake.date_between(start_date="+1y", end_date="+30y").isoformat(),
            "remaining_balance": round(loan_amount * random.uniform(0.1, 0.9), 2),
            "payment_status": random.choice(["Current", "Current", "Current", "Late", "Default"]),
            "last_payment_date": fake.date_between(start_date="-1m", end_date="today").isoformat(),
            "last_payment_amount": round(monthly_payment, 2),
            "collateral": fake.sentence() if random.choice([True, False]) else None,
        }

    def _generate_investment(self, **kwargs) -> Dict[str, Any]:
        """Generate an investment record."""
        purchase_date = fake.date_between(start_date="-5y", end_date="-1d")
        purchase_price = round(random.uniform(10, 500), 2)
        current_price = purchase_price * random.uniform(0.5, 2.0)
        quantity = random.randint(1, 1000)

        return {
            "investment_id": f"INV{random.randint(100000, 999999)}",
            "account_id": f"ACC{random.randint(100000, 999999)}",
            "investment_type": random.choice(self.INVESTMENT_TYPES),
            "symbol": fake.lexify(text="????").upper(),
            "name": fake.company(),
            "quantity": quantity,
            "purchase_date": purchase_date.isoformat(),
            "purchase_price": purchase_price,
            "current_price": round(current_price, 2),
            "total_value": round(current_price * quantity, 2),
            "cost_basis": round(purchase_price * quantity, 2),
            "unrealized_gain_loss": round((current_price - purchase_price) * quantity, 2),
            "dividend_yield": round(random.uniform(0, 5), 2) if random.choice([True, False]) else 0,
            "last_dividend_date": fake.date_between(start_date="-3m", end_date="today").isoformat()
            if random.choice([True, False])
            else None,
            "asset_allocation": random.choice(["Stocks", "Bonds", "Cash", "Alternative"]),
            "risk_level": random.choice(["Low", "Medium", "High"]),
        }

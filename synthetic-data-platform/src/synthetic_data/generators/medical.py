"""Medical domain synthetic data generator."""

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from faker import Faker

from ..core.generator import SyntheticDataGenerator

fake = Faker()


class MedicalDataGenerator(SyntheticDataGenerator):
    """Generator for medical/healthcare synthetic data."""

    # Medical data templates
    CONDITIONS = [
        "Hypertension",
        "Type 2 Diabetes",
        "Asthma",
        "COPD",
        "Arthritis",
        "Depression",
        "Anxiety",
        "Migraine",
        "Allergies",
        "Back Pain",
    ]

    MEDICATIONS = [
        "Lisinopril",
        "Metformin",
        "Albuterol",
        "Atorvastatin",
        "Omeprazole",
        "Ibuprofen",
        "Acetaminophen",
        "Amoxicillin",
        "Prednisone",
        "Losartan",
    ]

    PROCEDURES = [
        "Blood Test",
        "X-Ray",
        "MRI",
        "CT Scan",
        "Ultrasound",
        "EKG",
        "Physical Exam",
        "Vaccination",
        "Biopsy",
        "Endoscopy",
    ]

    DEPARTMENTS = [
        "Emergency",
        "Cardiology",
        "Neurology",
        "Orthopedics",
        "Pediatrics",
        "Oncology",
        "Radiology",
        "Surgery",
        "Internal Medicine",
        "Psychiatry",
    ]

    def generate_single(self, record_type: str = "patient", **kwargs) -> Dict[str, Any]:
        """
        Generate a single medical record.

        Args:
            record_type: Type of medical record (patient, encounter, lab_result, prescription)
            **kwargs: Additional generation parameters

        Returns:
            Generated medical record
        """
        if record_type == "patient":
            return self._generate_patient_record(**kwargs)
        elif record_type == "encounter":
            return self._generate_encounter_record(**kwargs)
        elif record_type == "lab_result":
            return self._generate_lab_result(**kwargs)
        elif record_type == "prescription":
            return self._generate_prescription(**kwargs)
        else:
            return self._generate_patient_record(**kwargs)

    def _generate_patient_record(self, **kwargs) -> Dict[str, Any]:
        """Generate a patient record."""
        dob = fake.date_of_birth(minimum_age=18, maximum_age=90)

        return {
            "patient_id": f"PAT{random.randint(100000, 999999)}",
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "date_of_birth": dob.isoformat(),
            "age": (datetime.now().date() - dob).days // 365,
            "gender": random.choice(["Male", "Female", "Other"]),
            "blood_type": random.choice(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]),
            "phone": fake.phone_number(),
            "email": fake.email(),
            "address": fake.street_address(),
            "city": fake.city(),
            "state": fake.state_abbr(),
            "zip_code": fake.postcode(),
            "insurance_provider": random.choice(
                ["BlueCross", "Aetna", "UnitedHealth", "Cigna", "Medicare"]
            ),
            "primary_condition": random.choice(self.CONDITIONS),
            "allergies": random.sample(
                ["Penicillin", "Peanuts", "Latex", "None"], k=random.randint(0, 2)
            ),
            "emergency_contact": fake.name(),
            "emergency_phone": fake.phone_number(),
            "registration_date": fake.date_between(start_date="-5y", end_date="today").isoformat(),
        }

    def _generate_encounter_record(self, **kwargs) -> Dict[str, Any]:
        """Generate a medical encounter record."""
        encounter_date = fake.date_time_between(start_date="-1y", end_date="now")

        return {
            "encounter_id": f"ENC{random.randint(100000, 999999)}",
            "patient_id": f"PAT{random.randint(100000, 999999)}",
            "encounter_date": encounter_date.isoformat(),
            "encounter_type": random.choice(["Routine", "Emergency", "Follow-up", "Consultation"]),
            "department": random.choice(self.DEPARTMENTS),
            "provider_name": f"Dr. {fake.last_name()}",
            "provider_id": f"DOC{random.randint(1000, 9999)}",
            "chief_complaint": random.choice(
                [
                    "Chest pain",
                    "Shortness of breath",
                    "Headache",
                    "Abdominal pain",
                    "Back pain",
                    "Fever",
                    "Cough",
                    "Fatigue",
                    "Dizziness",
                ]
            ),
            "diagnosis": random.choice(self.CONDITIONS),
            "procedures": random.sample(self.PROCEDURES, k=random.randint(1, 3)),
            "vital_signs": {
                "blood_pressure": f"{random.randint(110, 140)}/{random.randint(70, 90)}",
                "heart_rate": random.randint(60, 100),
                "temperature": round(random.uniform(97.0, 99.5), 1),
                "respiratory_rate": random.randint(12, 20),
                "oxygen_saturation": random.randint(95, 100),
            },
            "notes": fake.paragraph(),
            "follow_up_required": random.choice([True, False]),
            "follow_up_date": (encounter_date + timedelta(days=random.randint(7, 30)))
            .date()
            .isoformat()
            if random.choice([True, False])
            else None,
        }

    def _generate_lab_result(self, **kwargs) -> Dict[str, Any]:
        """Generate a lab result record."""
        test_date = fake.date_time_between(start_date="-6m", end_date="now")

        lab_tests = {
            "Complete Blood Count": {
                "WBC": (random.uniform(4.5, 11.0), "K/uL"),
                "RBC": (random.uniform(4.5, 5.9), "M/uL"),
                "Hemoglobin": (random.uniform(12.0, 17.5), "g/dL"),
                "Hematocrit": (random.uniform(36.0, 50.0), "%"),
                "Platelets": (random.uniform(150, 400), "K/uL"),
            },
            "Basic Metabolic Panel": {
                "Glucose": (random.uniform(70, 110), "mg/dL"),
                "Sodium": (random.uniform(136, 145), "mmol/L"),
                "Potassium": (random.uniform(3.5, 5.0), "mmol/L"),
                "Chloride": (random.uniform(98, 107), "mmol/L"),
                "CO2": (random.uniform(22, 29), "mmol/L"),
                "BUN": (random.uniform(7, 20), "mg/dL"),
                "Creatinine": (random.uniform(0.6, 1.2), "mg/dL"),
            },
            "Lipid Panel": {
                "Total Cholesterol": (random.uniform(125, 200), "mg/dL"),
                "LDL": (random.uniform(50, 130), "mg/dL"),
                "HDL": (random.uniform(40, 60), "mg/dL"),
                "Triglycerides": (random.uniform(50, 150), "mg/dL"),
            },
        }

        test_type = random.choice(list(lab_tests.keys()))
        results = {}
        for test, (value, unit) in lab_tests[test_type].items():
            results[test] = {
                "value": round(value, 2),
                "unit": unit,
                "status": random.choice(["Normal", "Normal", "Normal", "High", "Low"]),
            }

        return {
            "lab_id": f"LAB{random.randint(100000, 999999)}",
            "patient_id": f"PAT{random.randint(100000, 999999)}",
            "encounter_id": f"ENC{random.randint(100000, 999999)}",
            "test_date": test_date.isoformat(),
            "test_type": test_type,
            "ordering_provider": f"Dr. {fake.last_name()}",
            "lab_technician": fake.name(),
            "results": results,
            "status": "Completed",
            "critical_values": random.choice([True, False, False, False]),
            "notes": fake.sentence() if random.choice([True, False]) else None,
        }

    def _generate_prescription(self, **kwargs) -> Dict[str, Any]:
        """Generate a prescription record."""
        prescription_date = fake.date_between(start_date="-6m", end_date="today")

        return {
            "prescription_id": f"RX{random.randint(100000, 999999)}",
            "patient_id": f"PAT{random.randint(100000, 999999)}",
            "encounter_id": f"ENC{random.randint(100000, 999999)}",
            "medication_name": random.choice(self.MEDICATIONS),
            "generic_name": fake.word(),
            "dosage": f"{random.choice([5, 10, 20, 25, 50, 100])}mg",
            "frequency": random.choice(
                ["Once daily", "Twice daily", "Three times daily", "As needed"]
            ),
            "route": random.choice(["Oral", "Topical", "Injection", "Inhalation"]),
            "quantity": random.randint(30, 90),
            "refills": random.randint(0, 5),
            "prescriber": f"Dr. {fake.last_name()}",
            "prescriber_id": f"DOC{random.randint(1000, 9999)}",
            "pharmacy": f"{fake.company()} Pharmacy",
            "prescription_date": prescription_date.isoformat(),
            "start_date": prescription_date.isoformat(),
            "end_date": (prescription_date + timedelta(days=random.randint(30, 180))).isoformat(),
            "instructions": random.choice(
                [
                    "Take with food",
                    "Take on empty stomach",
                    "Take at bedtime",
                    "Take in the morning",
                    None,
                ]
            ),
            "warnings": random.choice(
                [
                    "May cause drowsiness",
                    "Avoid alcohol",
                    "Do not drive",
                    "May cause dizziness",
                    None,
                ]
            ),
        }

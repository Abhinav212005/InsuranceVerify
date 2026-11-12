"""Which fields are extracted for each document category."""
from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    kind: str  # text | name | date | amount | id
    mandatory: bool = True


FIELDS: dict[str, list[FieldSpec]] = {
    "insurance_policy": [
        FieldSpec("customer_name", "Customer name", "name"),
        FieldSpec("policy_number", "Policy number", "id"),
        FieldSpec("insurance_company", "Insurance company", "text"),
        FieldSpec("policy_start_date", "Policy start date", "date"),
        FieldSpec("policy_end_date", "Policy expiry date", "date"),
        FieldSpec("premium_amount", "Premium amount", "amount"),
        FieldSpec("coverage_amount", "Coverage amount", "amount", mandatory=False),
    ],
    "claim_form": [
        FieldSpec("claim_number", "Claim number", "id"),
        FieldSpec("policy_number", "Policy number", "id"),
        FieldSpec("customer_name", "Customer name", "name"),
        FieldSpec("incident_date", "Incident date", "date"),
        FieldSpec("claim_amount", "Claim amount", "amount"),
        FieldSpec("claim_description", "Claim description", "text", mandatory=False),
    ],
    "medical_bill": [
        FieldSpec("hospital_name", "Hospital name", "text"),
        FieldSpec("patient_name", "Patient name", "name"),
        FieldSpec("invoice_number", "Invoice number", "id"),
        FieldSpec("invoice_date", "Invoice date", "date"),
        FieldSpec("total_amount", "Total amount", "amount"),
    ],
    "repair_invoice": [
        FieldSpec("garage_name", "Garage name", "text"),
        FieldSpec("vehicle_details", "Vehicle details", "text"),
        FieldSpec("invoice_number", "Invoice number", "id"),
        FieldSpec("repair_date", "Repair date", "date"),
        FieldSpec("total_repair_cost", "Total repair cost", "amount"),
    ],
    "identity_document": [
        FieldSpec("document_type", "Document type", "text"),
        FieldSpec("name", "Name", "name"),
        FieldSpec("id_number", "ID number", "id"),
        FieldSpec("date_of_birth", "Date of birth", "date", mandatory=False),
        FieldSpec("father_name", "Father's name", "name", mandatory=False),
        FieldSpec("expiry_date", "Expiry date", "date", mandatory=False),
    ],
    "other": [],
}


def specs(category: str | None) -> list[FieldSpec]:
    return FIELDS.get(category or "other", [])


def schema_payload() -> dict:
    return {
        c: [{"name": s.name, "label": s.label, "kind": s.kind, "mandatory": s.mandatory} for s in specs_]
        for c, specs_ in FIELDS.items()
    }

"""Weighted keyword and pattern dictionaries for rule-based classification.

Weights: 5 = (almost) only appears in this kind of document, 3 = typical, 1 = weak hint.
Negative keywords subtract score when a phrase from a *different* document type shows up.
Regex patterns catch identifiers whose format is characteristic (PAN, Aadhaar, vehicle plates).
"""

CATEGORIES = [
    "insurance_policy",
    "claim_form",
    "identity_document",
    "medical_bill",
    "repair_invoice",
    "other",
]

KEYWORDS: dict[str, dict[str, float]] = {
    "insurance_policy": {
        "policy schedule": 5, "certificate of insurance": 5, "policy certificate": 5,
        "schedule of insurance": 5, "period of insurance": 5, "sum insured": 5,
        "policy period": 5, "insurance policy": 4, "policyholder": 4, "policy holder": 4,
        "policy wording": 5, "insured declared value": 4, "cover note": 4,
        "policy number": 3, "policy no": 3, "premium": 3, "insured": 3, "coverage": 3,
        "renewal": 3, "date of expiry": 3, "expiry date": 3, "inception date": 3,
        "deductible": 3, "exclusions": 3, "insurer": 3, "nominee": 3, "no claim bonus": 3,
        "idv": 2, "endorsement": 2, "valid from": 1, "valid until": 1, "insurance": 1, "policy": 1,
    },
    "claim_form": {
        "claim form": 5, "claim number": 5, "claim no": 5, "claim reference": 5,
        "notice of loss": 5, "claimant": 5, "date of loss": 5, "date of incident": 5,
        "incident date": 5, "nature of claim": 5, "details of the claim": 5,
        "claim intimation": 5, "claim amount": 5, "amount claimed": 5,
        "claim for reimbursement": 5, "reimbursement claim": 5, "description of incident": 5, "description of loss": 5,
        "declaration by claimant": 5, "first notice of loss": 5, "loss report": 4,
        "claimed amount": 4, "claim details": 4, "reporting party": 3,
        "witness": 3, "police report": 3, "fir number": 3, "circumstances of the accident": 4,
        "cause of loss": 3, "i hereby declare": 3, "place of incident": 3, "estimated loss": 3,
        "policy number": 2, "policy no": 2, "claim": 1, "incident": 1, "accident": 1,
    },
    "identity_document": {
        "date of birth": 5, "dob": 4, "aadhaar": 5, "unique identification": 5,
        "permanent account number": 5, "income tax department": 5, "passport": 5,
        "driving licence": 5, "driving license": 5, "voter id": 5, "election commission": 5,
        "identity card": 5, "national id": 5, "republic of india": 4, "government of india": 4,
        "father's name": 5, "date of issue": 3, "nationality": 4, "place of birth": 4,
        "valid upto": 3, "date of expiry": 1, "signature": 1, "male": 1, "female": 1,
        "photo": 1, "address": 1,
    },
    "medical_bill": {
        "hospital": 4, "patient name": 5, "patient": 3, "admission date": 5,
        "discharge date": 5, "date of admission": 5, "date of discharge": 5,
        "room charges": 5, "consultation fee": 5, "consultation charges": 5, "doctor": 3,
        "diagnosis": 4, "bed charges": 5, "pharmacy": 4, "laboratory charges": 5,
        "inpatient": 4, "uhid": 5, "ward": 2, "medical bill": 5, "hospital bill": 5,
        "final bill": 3, "nursing charges": 5, "operation theatre": 5, "ot charges": 5,
        "treating doctor": 5, "investigation charges": 4, "medicine charges": 4, "surgery": 3,
        "clinic": 3, "medical": 2, "diagnostic": 3, "radiology": 3, "pathology": 3, "icd": 3,
        "bill no": 1, "invoice": 1, "total amount": 1,
    },
    "repair_invoice": {
        "garage": 5, "workshop": 4, "service invoice": 5, "repair invoice": 5,
        "labour charges": 5, "labor charges": 5, "spare parts": 5, "parts total": 4,
        "vehicle number": 5, "registration number": 4, "reg no": 4, "chassis": 5,
        "make and model": 4, "odometer": 5, "job card": 5, "estimate for repair": 5,
        "body shop": 5, "denting": 5, "painting": 3, "wheel alignment": 5,
        "motors": 3, "auto works": 5, "automobile": 3, "mechanic": 4, "service center": 4,
        "service centre": 4, "labour": 3, "parts": 2, "vehicle": 3, "gst": 1, "hsn": 2,
        "invoice": 1, "km": 1,
    },
}

# phrases from other document types that count against a category
NEGATIVE: dict[str, dict[str, float]] = {
    "insurance_policy": {"claim form": 3, "patient name": 3, "labour charges": 3, "notice of loss": 3},
    "claim_form": {"policy schedule": 3, "certificate of insurance": 3, "labour charges": 2, "room charges": 2},
    "identity_document": {"premium": 3, "claim form": 3, "invoice": 3, "total amount": 3},
    "medical_bill": {"claim form": 3, "policy schedule": 3, "labour charges": 3, "chassis": 3},
    "repair_invoice": {"patient name": 4, "hospital": 3, "claim form": 3, "policy schedule": 3},
}

# (regex, weight, label) - applied to the ORIGINAL-case text
PATTERNS: dict[str, list[tuple[str, float, str]]] = {
    "identity_document": [
        (r"\b\d{4}\s\d{4}\s\d{4}\b", 5, "12-digit Aadhaar-style number"),
        (r"\b[A-Z]{5}\d{4}[A-Z]\b", 5, "PAN-style number"),
        (r"\b[A-Z]\d{7}\b", 2, "passport-style number"),
        (r"\b[A-Z]{2}\d{2}\s?\d{11}\b", 4, "driving-licence-style number"),
    ],
    "repair_invoice": [
        (r"\b[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4}\b", 3, "vehicle registration plate"),
    ],
    "claim_form": [
        (r"(?i)\bclaim\s*(?:no|number|ref)\.?\s*[:#-]?\s*[A-Z0-9][A-Z0-9/-]{3,}", 4, "claim number pattern"),
    ],
    "insurance_policy": [
        (r"(?i)\bpolicy\s*(?:no|number)\.?\s*[:#-]?\s*[A-Z0-9][A-Z0-9/-]{5,}", 2, "policy number pattern"),
    ],
    "medical_bill": [
        (r"(?i)\b(?:ip|uhid|mrn)\s*(?:no|number)?\.?\s*[:#-]?\s*[A-Z0-9-]{4,}", 3, "hospital registration number"),
    ],
}

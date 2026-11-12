"""Expected field values for tests/handwritten_docs.py (same order), written by reading each
document, not by looking at extractor output. None = the document does not state the field.
Where the honest answer is something the rules cannot reasonably read (a name buried in a
sentence), the true value is still listed, so those count as misses."""

TRUTH: list[dict] = [
    # ---- insurance_policy
    {"customer_name": "Sanjay Kulkarni", "policy_number": "ZM/PC/2025/00418822", "insurance_company": "Zenith Motor Insurance Ltd",
     "policy_start_date": "2025-02-14", "policy_end_date": "2026-02-13", "premium_amount": "15618.00", "coverage_amount": "610000.00"},
    {"customer_name": "Rekha Nambiar", "policy_number": None, "insurance_company": None,
     "policy_start_date": None, "policy_end_date": None, "premium_amount": "21400.00", "coverage_amount": "1000000.00"},
    {"customer_name": "Ghosh", "policy_number": "HP-2025-778120", "insurance_company": "Harbor Home Shield",
     "policy_start_date": None, "policy_end_date": None, "premium_amount": "6725.00", "coverage_amount": "4500000.00"},
    {"customer_name": "Anita Deshmukh", "policy_number": None, "insurance_company": None,
     "policy_start_date": "2025-07-01", "policy_end_date": None, "premium_amount": "14950.00", "coverage_amount": "10000000.00"},
    # ---- claim_form
    {"claim_number": "HC/2025/55012", "policy_number": "MED-7731-2024", "customer_name": "Imran Sheikh",
     "incident_date": None, "claim_amount": "85400.00", "claim_description": None},
    {"claim_number": None, "policy_number": "ZM/PC/2025/00418822", "customer_name": None,
     "incident_date": "2025-05-19", "claim_amount": "38000.00", "claim_description": "another car reversed into my parked vehicle."},
    {"claim_number": None, "policy_number": None, "customer_name": "Deepa Menon",
     "incident_date": "2025-07-11", "claim_amount": "45000.00", "claim_description": None},
    {"claim_number": None, "policy_number": "FIRE/2024/9981", "customer_name": "Bharat Traders",
     "incident_date": "2025-02-02", "claim_amount": "620000.00", "claim_description": "electrical short circuit in the godown"},
    # ---- identity_document
    {"document_type": "Voter ID", "name": "Salma Begum", "id_number": "XYZ4412907", "date_of_birth": "1984-08-12",
     "father_name": "Abdul Rahim", "expiry_date": None},
    {"document_type": "Aadhaar Card", "name": "Rohit Pawar", "id_number": "9143 2288 7716", "date_of_birth": "1993-11-21",
     "father_name": "Dinesh Pawar", "expiry_date": None},
    {"document_type": "Driving Licence", "name": "Kiran Joshi", "id_number": "MH12 20120045671", "date_of_birth": "1990-01-02",
     "father_name": "Mohan Joshi", "expiry_date": "2032-06-14"},
    {"document_type": "National ID Card", "name": "Marco Antonio Fernandes", "id_number": "12345678 9 ZZ4", "date_of_birth": "1988-03-17",
     "father_name": None, "expiry_date": "2030-03-16"},
    # ---- medical_bill (invoice_date: the bill prints no date; discharge date is the convention)
    {"hospital_name": "Lakeview Hospital Pvt Ltd", "patient_name": "Imran Sheikh", "invoice_number": None,
     "invoice_date": "2025-03-08", "total_amount": "85400.00"},
    {"hospital_name": "Sunshine Medicals & Pharmacy", "patient_name": "Meena Rao", "invoice_number": "33912",
     "invoice_date": None, "total_amount": "577.80"},
    {"hospital_name": "Care Plus Clinic", "patient_name": "Arun Nair", "invoice_number": None,
     "invoice_date": None, "total_amount": "750.00"},
    {"hospital_name": "Precision Diagnostics", "patient_name": "Kavita Soni", "invoice_number": None,
     "invoice_date": None, "total_amount": "8250.00"},
    # ---- repair_invoice
    {"garage_name": "Autocare Service Station", "vehicle_details": "Hyundai Creta, MH14CD7788", "invoice_number": "2291",
     "repair_date": None, "total_repair_cost": "7256.00"},
    {"garage_name": "Collision Repair Centre", "vehicle_details": "Maruti Baleno, KA03MN1209", "invoice_number": None,
     "repair_date": None, "total_repair_cost": "31152.00"},
    {"garage_name": "Metro Tyres & Battery", "vehicle_details": "DL8CAB5521", "invoice_number": None,
     "repair_date": None, "total_repair_cost": "34200.00"},
    {"garage_name": None, "vehicle_details": "Royal Enfield Classic 350, MH12KZ0098", "invoice_number": None,
     "repair_date": None, "total_repair_cost": "2070.00"},
    # ---- other (no fields)
    {}, {}, {}, {},
]

"""Synthetic OCR-style text corpus for training and evaluating classifiers AND extractors.

All people, companies, numbers and identifiers are invented. Every category has four
layout templates (A-D) with deliberately different wording, so a model can be trained on
A-C and tested on the unseen template D (template-held-out evaluation). Each sample also
carries `truth`, the expected extracted fields (None = the document does not contain it),
so field-extraction accuracy can be measured. Optional OCR noise (character confusions,
drops, duplicates) mimics real Tesseract mistakes; truth always stays clean.
"""
import random
import string
from dataclasses import dataclass

FIRST = ["Aarav", "Priya", "Rohan", "Sneha", "Vikram", "Ananya", "Karan", "Meera", "Arjun", "Divya",
         "Rahul", "Kavya", "Nikhil", "Isha", "Manish", "Pooja", "Sandeep", "Neha", "Amit", "Ritu"]
LAST = ["Sharma", "Verma", "Gupta", "Iyer", "Reddy", "Khan", "Patel", "Singh", "Nair", "Das",
        "Mehta", "Joshi", "Kulkarni", "Bose", "Rao", "Chopra", "Malhotra", "Pillai", "Sinha", "Thakur"]
INSURERS = ["Northstar General Insurance", "Bluepeak Assurance", "Sunrise Mutual Insurance", "Evergreen Life and General",
            "Harbor Shield Insurance", "Crestline Insurance Co.", "Maple Trust Assurance", "Unity First Insurance"]
HOSPITALS = ["City Care Hospital", "Sunrise Multispeciality Hospital", "Lotus Medical Centre", "Greenfield Hospital",
             "St. Marks Hospital", "Apex Heart Institute", "Riverside Hospital", "Hope General Hospital"]
CLINICS = ["MedLife Diagnostics", "Precision Pathology Lab", "Care Plus Clinic", "Vital Scan Centre"]
GARAGES = ["Ravi", "Speedway", "Shree Ganesh", "Prime", "Metro", "Highway King", "Star", "Royal"]
CITIES = ["Pune", "Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Kolkata", "Jaipur", "Lucknow", "Indore"]
CARS = ["Honda City", "Maruti Swift", "Hyundai i20", "Tata Nexon", "Toyota Innova", "Mahindra XUV300", "Kia Seltos"]
DESCRIPTIONS = [
    "Rear-end collision at a traffic signal, bumper and boot damaged",
    "Water entered the ground floor during heavy rain damaging furniture",
    "Fire in the kitchen caused by a short circuit, appliances destroyed",
    "Vehicle hit by an unknown truck while parked, driver side door dented",
    "Theft of laptop and bag from parked car, window glass broken",
]
DIAGNOSES = ["acute appendicitis", "viral fever with dehydration", "fractured tibia", "dengue fever", "cardiac arrhythmia"]
SURGERIES = ["appendectomy", "knee arthroscopy", "hernia repair", "angioplasty"]
STATES = ["MH", "DL", "KA", "TN", "TS", "WB", "RJ", "UP", "MP"]


@dataclass
class Sample:
    text: str
    label: str
    template: str
    truth: dict | None = None  # expected extracted fields (normalised)


# ---- value helpers: each returns (printed text, normalised truth) where both are needed ----
def _name(r): return f"{r.choice(FIRST)} {r.choice(LAST)}"


def _amt(r, lo=2000, hi=900000):
    n = r.randrange(lo, hi, 50)
    return (f"{n:,}" if r.random() < 0.5 else f"{n}.00"), f"{n:.2f}"


def _amount(r, lo=2000, hi=900000): return _amt(r, lo, hi)[0]


def _dt(r, y0=2023, y1=2026):
    d, m, y = r.randint(1, 28), r.randint(1, 12), r.randint(y0, y1)
    fmt = r.choice(["{d:02d}/{m:02d}/{y}", "{d:02d}-{m:02d}-{y}", "{y}-{m:02d}-{d:02d}"])
    return fmt.format(d=d, m=m, y=y), f"{y}-{m:02d}-{d:02d}"


def _date(r, y0=2023, y1=2026): return _dt(r, y0, y1)[0]
def _code(r, prefix, n=6): return f"{prefix}-{r.randint(2023, 2026)}-" + "".join(r.choice(string.digits) for _ in range(n))
def _reg(r): return f"{r.choice(STATES)}{r.randint(1, 49):02d}{''.join(r.choice(string.ascii_uppercase) for _ in range(2))}{r.randint(1000, 9999)}"
def _pan(r): return "".join(r.choice(string.ascii_uppercase) for _ in range(5)) + str(r.randint(1000, 9999)) + r.choice(string.ascii_uppercase)
def _aadhaar(r): return " ".join(str(r.randint(1000, 9999)) for _ in range(3))
def _line_items(r, labels): return "\n".join(f"{x}  {_amount(r, 200, 40000)}" for x in labels)


def policy(r, t):
    co, name, pol = r.choice(INSURERS), _name(r), _code(r, "POL")
    (d1, i1), (d2, i2) = _dt(r), _dt(r)
    (amt, namt), (prem, nprem) = _amt(r, 100000, 5000000), _amt(r, 3000, 60000)
    truth = {"customer_name": name, "policy_number": pol, "insurance_company": co, "policy_start_date": i1,
             "policy_end_date": i2, "premium_amount": nprem, "coverage_amount": namt}
    if t == "A":
        text = (f"{co}\nPOLICY SCHEDULE\nPolicy No: {pol}\nPolicyholder: {name}\nPeriod of Insurance: {d1} to {d2}\n"
                f"Sum Insured: Rs. {amt}\nPremium: Rs. {prem}\nThis schedule forms part of your policy and should be read with the policy wording.")
    elif t == "B":
        truth.update(insurance_company=f"{co} Co. Ltd", premium_amount=None)
        text = (f"{co} Co. Ltd.\nCertificate of Insurance\nPolicy Number {pol}\nInsured Name {name}\nValid from {d1} valid until {d2}\n"
                f"Coverage {amt}\nDeductible Rs. {_amount(r, 500, 5000)}\nExclusions apply as listed in the policy wording.")
    elif t == "C":
        truth["policy_number"] = None
        text = (f"MOTOR INSURANCE POLICY\nInsurer: {co}\nInsured: {name}\nVehicle {_reg(r)} {r.choice(CARS)}\nInsured Declared Value Rs {amt}\n"
                f"Policy period {d1} - {d2}\nNo claim bonus {r.choice([20, 25, 35, 50])}%\nTotal premium payable {prem}")
    else:
        truth["policy_start_date"] = None
        text = (f"Your Health Cover - Renewal Notice\n{co}\nDear {name},\nYour cover under policy {pol} expires on {d2}. "
                f"Annual premium for renewal: Rs {prem}. Sum assured {amt}. Nominee: {_name(r)}. Please renew to continue your coverage.")
    return text, truth


def claim(r, t):
    co, name, pol, clm = r.choice(INSURERS), _name(r), _code(r, "POL"), _code(r, "CLM", 5)
    (d1, i1), (amt, namt), desc, city = _dt(r), _amt(r, 5000, 800000), r.choice(DESCRIPTIONS), r.choice(CITIES)
    truth = {"claim_number": clm, "policy_number": pol, "customer_name": name, "incident_date": i1,
             "claim_amount": namt, "claim_description": desc}
    if t == "A":
        text = (f"CLAIM FORM\nClaim No: {clm}\nPolicy Number: {pol}\nClaimant Name: {name}\nDate of Incident: {d1}\n"
                f"Claim Amount: Rs. {amt}\nDescription of Incident: {desc}\nI hereby declare that the information given is true.")
    elif t == "B":
        truth["claim_description"] = None
        text = (f"{co}\nNotice of Loss\nClaim Reference {clm}\nPolicy No {pol}\nInsured {name}\nDate of loss {d1}\nPlace of incident {city}\n"
                f"Estimated loss {amt}\nPolice report / FIR number {r.randint(100, 999)}/{r.randint(2023, 2026)}")
    elif t == "C":
        truth.update(claim_number=None, policy_number=None, incident_date=None)
        text = (f"Motor Claim Intimation\nClaimant: {name}\nVehicle {_reg(r)}\nCircumstances of the accident: {desc}\n"
                f"Amount claimed Rs {amt}\nWitness: {_name(r)}\nDeclaration by claimant: all details are correct. Signature")
    else:
        truth["policy_number"] = None
        text = (f"First Notice of Loss - Reporting Party: {name}\nReference: {clm}\nWhat happened: {desc}\nWhen: {d1}  Where: {city}\n"
                f"Loss report estimate {amt}\nContact the claims desk for updates.")
    return text, truth


def identity(r, t):
    name, (dob, idob) = _name(r), _dt(r, 1955, 2004)
    if t == "A":
        fname, pan = _name(r), _pan(r)
        text = (f"INCOME TAX DEPARTMENT\nGOVT. OF INDIA\nPermanent Account Number\n{pan}\nName {name.upper()}\n"
                f"Father's Name {fname.upper()}\nDate of Birth {dob}\nSignature")
        truth = {"document_type": "PAN Card", "name": name, "id_number": pan, "date_of_birth": idob,
                 "father_name": fname, "expiry_date": None}
    elif t == "B":
        num = _aadhaar(r)
        text = f"Government of India\nUnique Identification Authority of India\n{name}\nDOB: {dob}\n{r.choice(['Male', 'Female'])}\n{num}\nAadhaar"
        truth = {"document_type": "Aadhaar Card", "name": name, "id_number": num, "date_of_birth": idob,
                 "father_name": None, "expiry_date": None}
    elif t == "C":
        dl = f"{r.choice(STATES)}{r.randint(10, 99)} {r.randint(10**10, 10**11 - 1)}"
        (d1, _), (d2, i2) = _dt(r), _dt(r, 2027, 2040)
        text = (f"DRIVING LICENCE\nDL No {dl}\nName {name}\nDate of Birth {dob}\n"
                f"Date of issue {d1}\nValid upto {d2}\nAddress {r.choice(CITIES)}\nSignature")
        truth = {"document_type": "Driving Licence", "name": name, "id_number": dl, "date_of_birth": idob,
                 "father_name": None, "expiry_date": i2}
    else:
        ppt = f"{r.choice(string.ascii_uppercase)}{r.randint(1000000, 9999999)}"
        text = (f"REPUBLIC OF INDIA\nPASSPORT\nType P Country Code IND\nPassport No. {ppt}\n"
                f"Surname {name.split()[1]}\nGiven names {name.split()[0]}\nNationality INDIAN\nDate of birth {dob}\n"
                f"Place of birth {r.choice(CITIES)}\nDate of issue {_date(r)}")
        truth = {"document_type": "Passport", "name": name, "id_number": ppt, "date_of_birth": idob,
                 "father_name": None, "expiry_date": None}
    return text, truth


def medical(r, t):
    name, hosp = _name(r), r.choice(HOSPITALS)
    amt, namt = _amt(r, 3000, 600000)
    (d1, _), (d2, i2) = _dt(r), _dt(r)
    # no bill date is printed on most hospital bills; the extractor falls back to the discharge date
    truth = {"hospital_name": hosp, "patient_name": name, "invoice_number": None, "invoice_date": i2, "total_amount": namt}
    if t == "A":
        text = (f"{hosp}\nFINAL BILL\nPatient Name: {name}\nUHID {r.randint(100000, 999999)}\nDate of Admission {d1}  Date of Discharge {d2}\n"
                f"{_line_items(r, ['Room Charges', 'Consultation Fee', 'Pharmacy'])}\nTotal Amount Rs. {amt}")
    elif t == "B":
        bill = _code(r, "B", 5)
        truth.update(invoice_number=bill, invoice_date=None)
        text = (f"{hosp} - In-patient Bill\nBill No {bill}\nPatient {name}\nTreating Doctor Dr. {_name(r)}\nDiagnosis {r.choice(DIAGNOSES)}\n"
                f"Bed charges, Nursing charges, Investigation charges, Medicine charges\nGrand total {amt}")
    elif t == "C":
        clinic, bill = r.choice(CLINICS), _code(r, "LAB", 5)
        truth.update(hospital_name=f"{clinic} Diagnostic and Pathology Laboratory", invoice_number=bill, invoice_date=None)
        text = (f"{clinic} Diagnostic and Pathology Laboratory\nInvoice {bill}\nPatient name {name}\n"
                f"Tests: CBC, LFT, X-ray (radiology)\nConsultation charges {_amount(r, 300, 2000)}\nTotal {amt}\nReferred by Dr. {_name(r)}")
    else:
        truth.update(hospital_name=None, patient_name=None)
        text = (f"Discharge Summary and Bill\nAdmitted {d1}, discharged {d2}\nWard: General\nSurgery performed: {r.choice(SURGERIES)}\n"
                f"Operation theatre charges {_amount(r, 10000, 90000)}\nMedical expenses total {amt}\nInsurance TPA: {r.choice(INSURERS)}")
    return text, truth


def repair(r, t):
    garage, car, reg = r.choice(GARAGES), r.choice(CARS), _reg(r)
    amt, namt = _amt(r, 1500, 150000)
    truth = {"garage_name": f"{garage} Motors", "vehicle_details": f"{car}, {reg}", "invoice_number": None,
             "repair_date": None, "total_repair_cost": namt}
    if t == "A":
        inv = _code(r, "INV", 5)
        truth["invoice_number"] = inv
        text = (f"{garage} Motors\nTAX INVOICE\nInvoice No {inv}\nVehicle Number {reg}\nMake and Model {car}\n"
                f"Odometer {r.randint(5000, 120000)} km\n{_line_items(r, ['Labour Charges', 'Spare Parts'])}\nGST 18% {_amount(r, 100, 9000)}\nTotal {amt}")
    elif t == "B":
        truth.update(garage_name=f"{garage} Auto Works", vehicle_details=reg)
        text = (f"{garage} Auto Works - Service Invoice\nJob card {r.randint(1000, 9999)}\nCustomer {_name(r)}\nReg no {reg}\n"
                f"Wheel alignment, denting and painting\nParts total {_amount(r, 500, 60000)}\nLabor charges {_amount(r, 500, 30000)}\nGrand total {amt}")
    elif t == "C":
        truth.update(garage_name=f"{garage} Body Shop")
        chassis = "".join(r.choice(string.ascii_uppercase + string.digits) for _ in range(12))
        text = (f"{garage} Body Shop\nEstimate for repair\nVehicle: {car} ({reg})\nChassis No {chassis}\n"
                f"Work: bumper replacement, painting\nLabour {_amount(r, 500, 20000)}  Parts {_amount(r, 500, 60000)}\nPayable {amt}\n"
                f"This estimate is to be submitted to your insurer.")
    else:
        truth.update(garage_name=garage)
        text = (f"Service Centre Bill\nCar: {car}\nPlate {reg}\nMechanic notes: oil change, brake pads, engine tune up\n"
                f"HSN 8708 parts {_amount(r, 500, 50000)}\nLabour {_amount(r, 500, 20000)}\nInvoice total {amt}  Thank you for visiting {garage}")
    return text, truth


def other(r, t):
    name = _name(r)
    if t == "A":
        text = (f"State Electricity Board\nBill for the month of {r.choice(['March', 'June', 'October'])}\nConsumer No {r.randint(10**8, 10**9)}\n"
                f"Name {name}\nUnits consumed {r.randint(80, 600)}\nAmount due Rs. {_amount(r, 300, 9000)}\nDue date {_date(r)}")
    elif t == "B":
        text = (f"Dear Sir/Madam,\nI am writing to request a copy of the insurance brochure and an appointment with your agent next week "
                f"to understand the premium options for my family.\nYours sincerely,\n{name}\n{r.choice(CITIES)}")
    elif t == "C":
        text = (f"{r.choice(['Spice Garden', 'Cafe Aroma', 'Tandoor Nights', 'Green Bowl'])} Restaurant\nTable {r.randint(1, 30)}\n"
                f"{_line_items(r, ['Paneer Tikka', 'Butter Naan', 'Cold Coffee'])}\nService charge 5%\nTotal Rs {_amount(r, 300, 4000)}\nThank you, visit again")
    else:
        text = (f"Account Statement\nAccount No {r.randint(10**9, 10**10)}\nOpening balance {_amount(r, 1000, 90000)}\n"
                f"Date  Narration  Debit  Credit  Balance\n{_date(r)}  UPI transfer  {_amount(r, 100, 5000)}  0.00  {_amount(r, 1000, 90000)}\n"
                f"{_date(r)}  Salary credit  0.00  {_amount(r, 20000, 90000)}  {_amount(r, 1000, 90000)}")
    return text, {}


GENERATORS = {
    "insurance_policy": policy,
    "claim_form": claim,
    "identity_document": identity,
    "medical_bill": medical,
    "repair_invoice": repair,
    "other": other,
}
TEMPLATES = ["A", "B", "C", "D"]

_CONFUSIONS = {"o": "0", "O": "0", "0": "O", "l": "1", "1": "l", "i": "l", "I": "l", "e": "c", "c": "e",
               "m": "rn", "S": "5", "5": "S", "B": "8", "8": "B", "a": "o", "u": "v", "n": "h"}


def ocr_noise(text: str, rate: float, rng: random.Random) -> str:
    """Corrupt characters like an OCR engine would: confusions, drops, duplicates."""
    if rate <= 0:
        return text
    out = []
    for ch in text:
        if ch.isalnum() and rng.random() < rate:
            op = rng.random()
            if op < 0.6 and ch in _CONFUSIONS:
                out.append(_CONFUSIONS[ch])
            elif op < 0.8:
                continue  # dropped
            else:
                out.append(ch + ch)  # duplicated
        else:
            out.append(ch)
    return "".join(out)


def generate(per_template: int = 30, templates: list[str] | None = None, noise: float = 0.0, seed: int = 0) -> list[Sample]:
    rng = random.Random(seed)
    samples: list[Sample] = []
    for label, gen in GENERATORS.items():
        for t in templates or TEMPLATES:
            for _ in range(per_template):
                text, truth = gen(rng, t)
                # real OCR text varies in noise level, so sample it per document
                level = rng.uniform(0, noise) if noise else 0.0
                samples.append(Sample(ocr_noise(text, level, rng), label, t, truth))
    rng.shuffle(samples)
    return samples

"""Independent, hand-written documents (not produced by the synthetic generator) with
deliberately different wording and hard cases such as a health policy that mentions
hospitals, a hospitalisation claim form and a repair estimate that mentions the insurer.
All names and numbers are invented. Used as a sanity benchmark: it was written once and
not tuned against."""

DOCS: list[tuple[str, str]] = [
    # ---- insurance_policy
    ("insurance_policy", """Zenith Motor Insurance Ltd.
Policy Schedule cum Certificate of Insurance (Private Car Package Policy)
Policy No. ZM/PC/2025/00418822      Registration No. MH 12 AB 4411
Insured: Mr Sanjay Kulkarni, Baner Road, Pune
Insured Declared Value (IDV): Rs. 6,10,000
Own Damage premium 9,820.00   Third party premium 3,416.00   GST 2,382.00
Total premium payable Rs. 15,618.00
Period of cover: 14/02/2025 00:00 hrs to 13/02/2026 midnight"""),
    ("insurance_policy", """FAMILY FLOATER HEALTH PLAN - POLICY DOCUMENT
Policy holder: Mrs Rekha Nambiar     Members covered: Self, spouse, 2 children
Sum Insured Rs. 10,00,000 per policy year
Cashless treatment is available at network hospitals. Waiting period for pre-existing
diseases is 36 months. Room rent limit 1% of sum insured. Renewal premium Rs. 21,400."""),
    ("insurance_policy", """Dear Mr Ghosh,
Thank you for choosing Harbor Home Shield. Your home insurance is now active.
Your policy number is HP-2025-778120. It covers the building and contents of your flat up to
Rs 45,00,000 against fire, flood and burglary. Annual premium paid: Rs 6,725.
Please read the policy wording and exclusions carefully."""),
    ("insurance_policy", """Term Life Insurance - Proposal accepted
Life assured: Anita Deshmukh     Nominee: Rahul Deshmukh
Policy commencement date 01/07/2025   Premium payment term 20 years
Sum assured Rs 1,00,00,000   Annual premium Rs 14,950   Policy term 35 years"""),
    # ---- claim_form
    ("claim_form", """Reimbursement claim - Hospitalisation
Claim ID: HC/2025/55012          Insured member: Mr Imran Sheikh
Policy no: MED-7731-2024
Date of admission 03/03/2025  Date of discharge 08/03/2025  Hospital: Lakeview Hospital
Total amount claimed Rs 85,400
Documents enclosed: hospital bills, discharge summary, prescriptions.
I declare that the above statements are true."""),
    ("claim_form", """VEHICLE DAMAGE REPORT (to be filled by the insured)
Policy no: ZM/PC/2025/00418822
Accident date: 19/05/2025     Accident location: Katraj Chowk, Pune
Describe how the accident happened: another car reversed into my parked vehicle.
Third party details: Not known. Police informed: Yes, FIR no 214/2025.
Estimated cost of repair Rs 38,000"""),
    ("claim_form", """THEFT CLAIM
I wish to claim for my mobile phone stolen from my bag on a train on 11 July 2025.
Claim number: to be allotted by the company
Copy of the FIR is attached. Value of the item claimed: Rs 45,000.
Name of claimant: Deepa Menon      Signature"""),
    ("claim_form", """INTIMATION OF LOSS - FIRE
Date and time of loss: 02/02/2025 around 11 pm
Cause of loss (as far as known): electrical short circuit in the godown
Name of claimant: Bharat Traders      Policy no FIRE/2024/9981
Sum claimed: Rs 6,20,000     Stock register and photographs enclosed."""),
    # ---- identity_document
    ("identity_document", """ELECTION COMMISSION OF INDIA
IDENTITY CARD
Elector's Name: Salma Begum
Father's Name: Abdul Rahim
Sex: Female    Date of Birth: 12/08/1984
EPIC No: XYZ4412907"""),
    ("identity_document", """Aadhaar e-card - Download date 04/04/2025
Enrolment No: 1234/56789/01234
To: Rohit Pawar, S/O Dinesh Pawar, Flat 4, Shivaji Nagar, Nashik
VID: 9143 2288 7716 0042
DOB: 21/11/1993    Male
9143 2288 7716"""),
    ("identity_document", """DRIVING LICENCE - UNION OF INDIA
Name: Kiran Joshi    S/D/W of: Mohan Joshi
DOB: 02/01/1990     Blood group: B+
Class of vehicle: LMV, MCWG
Issued by RTO Pune on 15/06/2012   Valid upto 14/06/2032
DL No: MH12 20120045671"""),
    ("identity_document", """NATIONAL IDENTITY CARD
Surname: Fernandes      Given names: Marco Antonio
Date of birth 17 MAR 1988    Sex M    Nationality: Portuguese
Card number 12345678 9 ZZ4    Date of expiry 16 MAR 2030"""),
    # ---- medical_bill
    ("medical_bill", """Lakeview Hospital Pvt Ltd
INPATIENT FINAL INVOICE
Patient: Mr Imran Sheikh   Age/Sex: 41/M      IP No: 88120
Admission 03/03/2025   Discharge 08/03/2025   Bed category: semi private
Doctor visits 14,000    OT charges 32,500    Pharmacy and consumables 11,870
Room rent (5 days) 17,500    Investigation charges 9,530
Net payable Rs 85,400"""),
    ("medical_bill", """Sunshine Medicals & Pharmacy
Bill no 33912     Patient name: Meena Rao      Prescribed by Dr. P. Kamath
Amoxicillin 500mg x 15   Paracetamol 650 x 10   Vitamin D3 x 4   ORS sachets x 6
Total MRP 642.00   Discount 10%   Amount payable 577.80"""),
    ("medical_bill", """CARE PLUS CLINIC - OPD receipt
Patient name: Arun Nair    Consultation with Dr. Sheela Thomas (Physician)
Consultation fee Rs 600    Dressing charges Rs 150
Total received Rs 750 - Thank you"""),
    ("medical_bill", """Precision Diagnostics - Invoice
Patient: Kavita Soni      Referred by: Dr Anil Shetty
MRI Brain (radiology) 6,800    Blood tests - CBC, Lipid profile 1,450
Total 8,250   Paid by card"""),
    # ---- repair_invoice
    ("repair_invoice", """AUTOCARE SERVICE STATION
Tax invoice no 2291      Vehicle reg: MH14 CD 7788      Model: Hyundai Creta
Services: engine oil change, oil filter, brake pad replacement (front)
Labour 1,800    Parts 4,350    CGST 9% 553    SGST 9% 553
Total payable Rs 7,256"""),
    ("repair_invoice", """ESTIMATE - COLLISION REPAIR CENTRE
Car: Maruti Baleno   Reg no KA03 MN 1209
Front bumper replace, headlamp assembly, bonnet repaint
Labour 7,500   Parts 18,900   GST 18% 4,752   Estimate total 31,152
Insurer approval pending - estimate valid for 7 days"""),
    ("repair_invoice", """Metro Tyres & Battery
Customer vehicle number DL 8C AB 5521
4 x tyre replacement, wheel balancing and alignment, new battery 65Ah
Total Rs 34,200  - 1 year warranty on battery"""),
    ("repair_invoice", """Two wheeler service bill
Bike: Royal Enfield Classic 350   Reg MH 12 KZ 0098   KM reading 18,420
Chain lubrication, oil change, brake adjustment   Labour 650   Spares 1,420
Total 2,070"""),
    # ---- other
    ("other", """RESIDENTIAL RENTAL AGREEMENT
This agreement is made on 1 April 2025 between the landlord Mr Prakash Iyer and the tenant Ms Tara Menon.
Monthly rent Rs 28,000 payable on or before the 5th of each month. Security deposit Rs 84,000.
The tenancy is for eleven months."""),
    ("other", """E-TICKET / BOOKING CONFIRMATION
PNR: QX7T2K   Passenger: Mr Vivek Anand
Flight AI 664  Pune - Delhi   Departure 14 Sep 2025 06:15   Arrival 08:20
Baggage 15 kg   Fare Rs 5,480"""),
    ("other", """Offer of Employment
Dear Neha, we are pleased to offer you the position of Software Engineer at Bright Labs.
Your annual CTC will be Rs 12,00,000. Date of joining: 01 August 2025.
Please sign and return a copy of this letter."""),
    ("other", """Greenfield Public School - Fee Receipt
Student: Aditya Rao   Class: 7B   Term: II
Tuition fee 18,000   Activity fee 2,500   Total received Rs 20,500"""),
]

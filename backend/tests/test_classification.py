import cv2
import pytest
from sqlalchemy import create_engine, inspect, text

from app.classification import classifier, keyword_classifier as kw, ml_classifier as ml
from app.classification.keywords import CATEGORIES
from app.classification.synthetic import GENERATORS, generate, ocr_noise
from app.core.config import settings
from app.core.errors import AppError
from tests.dip_helpers import make_page
from tests.handwritten_docs import DOCS

P = "/api/v1"

POLICY = "NORTHSTAR INSURANCE\nPOLICY SCHEDULE\nPolicy No: POL-2025-001234\nPolicyholder: Aarav Sharma\nPeriod of Insurance: 01/04/2025 to 31/03/2026\nSum Insured: Rs. 500000\nPremium: Rs. 12450"
CLAIM = "CLAIM FORM\nClaim No: CLM-2025-00042\nPolicy Number: POL-2025-001234\nClaimant Name: Aarav Sharma\nDate of Incident: 12/05/2025\nClaim Amount: Rs. 80000\nDescription of Incident: rear-end collision"
ID = "INCOME TAX DEPARTMENT\nGOVT. OF INDIA\nPermanent Account Number\nABCDE1234F\nName AARAV SHARMA\nFather's Name RAJESH SHARMA\nDate of Birth 01/01/1990"
MED = "City Care Hospital\nFINAL BILL\nPatient Name: Aarav Sharma\nDate of Admission 03/03/2025 Date of Discharge 08/03/2025\nRoom Charges 17500\nConsultation Fee 4000\nTotal Amount Rs. 85400"
REPAIR = "Ravi Motors\nTAX INVOICE\nVehicle Number MH12AB1234\nMake and Model Honda City\nLabour Charges 1800\nSpare Parts 4350\nTotal 7256"


@pytest.fixture(scope="module")
def model():
    train = generate(25, noise=0.05, seed=3) + generate(25, noise=0.0, seed=4)
    return ml.train([s.text for s in train], [s.label for s in train])


# ---- keyword classifier ------------------------------------------------------------
@pytest.mark.parametrize(
    "text,label",
    [(POLICY, "insurance_policy"), (CLAIM, "claim_form"), (ID, "identity_document"), (MED, "medical_bill"), (REPAIR, "repair_invoice")],
)
def test_keyword_classifies_clear_documents(text, label):
    r = kw.classify(text)
    assert r["category"] == label and r["confidence"] > 0.5
    assert r["evidence"] and all(e["weight"] > 0 for e in r["evidence"])
    assert abs(sum(r["probabilities"].values()) - 1) < 0.01


def test_unrelated_text_falls_back_to_other_with_low_confidence():
    r = kw.classify("Hello, the meeting is moved to Thursday at 4 pm. Please bring the quarterly slides.")
    assert r["category"] == "other"
    r2 = kw.classify("")
    assert r2["category"] == "other"
    # weak hints exactly equal to the "other" prior are a tie, which must resolve to other
    weak = kw.classify("Please find our insurance brochure and premium list.")
    assert weak["scores"]["insurance_policy"] == kw.OTHER_PRIOR and weak["category"] == "other"


def test_typo_tolerance_and_ocr_noise():
    noisy = POLICY.replace("POLICY SCHEDULE", "POLICY SCHEDULC").replace("Period of Insurance", "Perlod of lnsurance")
    r = kw.classify(noisy)
    assert r["category"] == "insurance_policy"
    assert any(e["fuzzy"] for e in r["evidence"]) or r["confidence"] > 0.4


def test_negative_keywords_and_hard_cases():
    # a repair estimate that mentions the insurer must stay a repair invoice
    est = "Metro Body Shop\nEstimate for repair\nLabour charges 5000\nSpare parts 9000\nVehicle MH12AB1234\nSubmit to your insurer for approval"
    assert kw.classify(est)["category"] == "repair_invoice"
    # a health policy that talks about hospitals must stay a policy
    health = "Health policy document\nPolicy holder: A Rao\nSum insured Rs 5,00,000\nCashless treatment at network hospitals\nPremium Rs 12,000"
    assert kw.classify(health)["category"] == "insurance_policy"


def test_regex_patterns_identify_identifiers():
    assert kw.classify("ABCDE1234F")["scores"]["identity_document"] >= 5  # PAN pattern alone
    assert kw.score("registration MH12AB1234")["scores"]["repair_invoice"] >= 3


def test_title_boost_prefers_headline_phrase():
    a = kw.score("CLAIM FORM\n" + "x " * 300)["scores"]["claim_form"]
    b = kw.score("x " * 300 + "\nCLAIM FORM")["scores"]["claim_form"]
    assert a > b


# ---- hybrid, ML, wiring ------------------------------------------------------------
def accuracy(method, model=None):
    ok = sum(classifier.classify(t, method, model)["category"] == label for label, t in DOCS)
    return ok / len(DOCS)


def test_benchmark_on_independent_handwritten_documents(model):
    """Written once, not tuned: guards against regressions and against overfitting to templates."""
    assert accuracy("keyword") >= 0.9
    assert accuracy("hybrid", model) >= 0.9


def test_ml_alone_is_weaker_on_unseen_wording(model):
    assert accuracy("ml", model) >= 0.6  # sanity floor; keyword/hybrid are the recommended methods


def test_hybrid_degrades_to_keyword_without_model(monkeypatch):
    monkeypatch.setattr(ml, "load", lambda *a, **k: None)
    r = classifier.classify(POLICY, "hybrid")
    assert r["method"] == "keyword" and "no trained" in r["note"]
    assert classifier.available_methods() == ["keyword"]


def test_invalid_method_and_empty_text():
    with pytest.raises(AppError) as e:
        classifier.classify("x", "magic")
    assert e.value.code == "INVALID_METHOD"
    r = classifier.classify("   ")
    assert r["category"] == "other" and r["confidence"] == 0.0


def test_hybrid_reports_both_opinions(model):
    r = classifier.classify(CLAIM, "hybrid", model)
    assert r["category"] == "claim_form" and r["keyword"]["category"] == "claim_form" and r["ml"]["category"]
    assert set(r["probabilities"]) == set(CATEGORIES)


def test_model_save_and_load_roundtrip(model, tmp_path):
    path = tmp_path / "m.joblib"
    ml.save(model, {"docs": 1}, path)
    ml.reset_cache()
    loaded = ml.load(path)
    assert loaded is not None and ml.metadata(path)["docs"] == 1
    assert ml.probabilities(loaded, MED)["medical_bill"] > 0.3
    assert ml.load(tmp_path / "missing.joblib") is None


def test_synthetic_corpus_is_reproducible_and_labelled():
    a, b = generate(3, seed=5), generate(3, seed=5)
    assert [s.text for s in a] == [s.text for s in b]
    assert len(a) == 3 * 4 * len(GENERATORS) and {s.label for s in a} == set(CATEGORIES)
    import random

    assert ocr_noise("Policy Number 12345", 0.5, random.Random(1)) != "Policy Number 12345"
    assert ocr_noise("Policy", 0.0, random.Random(1)) == "Policy"


# ---- API ---------------------------------------------------------------------------------
def png(img) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def upload(client, lines):
    page = make_page(lines=lines)
    r = client.post(f"{P}/documents", files=[("files", ("doc.png", png(page), "image/png"))])
    return r.json()["items"][0]["id"]


@pytest.fixture()
def full_pipeline(monkeypatch):
    monkeypatch.setattr(settings, "auto_ocr", True)
    monkeypatch.setattr(settings, "auto_classify", True)


def test_upload_is_classified_automatically(client, full_pipeline):
    doc_id = upload(client, ["CLAIM FORM", "Claim No: CLM-2025-00042", "Policy Number: POL-2025-001234",
                             "Claimant Name: Aarav Sharma", "Date of Incident: 12/05/2025", "Claim Amount: Rs. 80000"])
    d = client.get(f"{P}/documents/{doc_id}").json()
    assert d["doc_category"] == "claim_form" and d["category_confidence"] > 0.5
    assert d["classification_method"] in ("hybrid", "keyword")
    assert d["classification_json"]["evidence"] and d["classification_json"]["probabilities"]["claim_form"] > 0.5
    assert client.get(f"{P}/documents/{doc_id}/status").json()["stage"] == "done"


def test_reclassify_with_method_and_errors(client, full_pipeline):
    doc_id = upload(client, ["POLICY SCHEDULE", "Policy No: POL-2025-001234", "Period of Insurance: 01/04/2025 to 31/03/2026",
                             "Sum Insured: Rs. 500000", "Premium: Rs. 12450"])
    r = client.post(f"{P}/documents/{doc_id}/classify", json={"method": "keyword"})
    assert r.status_code == 200 and r.json()["category"] == "insurance_policy" and r.json()["method"] == "keyword"
    assert client.get(f"{P}/documents/{doc_id}").json()["classification_method"] == "keyword"
    assert client.post(f"{P}/documents/{doc_id}/classify", json={"method": "magic"}).status_code == 422
    assert client.post(f"{P}/documents/9999/classify", json={}).status_code == 404


def test_classify_without_text_is_refused(client):
    doc_id = upload(client, ["anything"])  # auto_ocr is off in tests, so no OCR text exists
    r = client.post(f"{P}/documents/{doc_id}/classify", json={})
    assert r.status_code == 409 and r.json()["error"]["code"] == "NO_TEXT"


def test_manual_override_is_recorded(client, full_pipeline):
    doc_id = upload(client, ["CLAIM FORM", "Claim No: CLM-1", "Claimant: A B", "Date of Incident: 12/05/2025"])
    r = client.patch(f"{P}/documents/{doc_id}/fields", json={"category": "other"}).json()
    assert r["doc_category"] == "other" and r["classification_method"] == "manual" and r["category_confidence"] == 1.0


def test_classification_info_endpoint(client):
    r = client.get(f"{P}/classification/info").json()
    assert r["categories"] == CATEGORIES and "keyword" in r["methods"]


# ---- schema migration -----------------------------------------------------------------------
def test_init_db_adds_missing_column_to_old_database(tmp_path):
    from app.db.session import init_db

    engine = create_engine(f"sqlite:///{(tmp_path / 'old.db').as_posix()}")
    with engine.begin() as c:  # a database created before classification_json existed
        c.execute(text("CREATE TABLE documents (id INTEGER PRIMARY KEY, filename TEXT)"))
    init_db(engine)
    assert "classification_json" in {c["name"] for c in inspect(engine).get_columns("documents")}
    init_db(engine)  # idempotent

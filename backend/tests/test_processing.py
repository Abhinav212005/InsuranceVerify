import cv2
import pytest

from tests.dip_helpers import add_gaussian_noise, make_page, rotate_img

P = "/api/v1"


@pytest.fixture()
def doc_id(client):
    page = add_gaussian_noise(rotate_img(make_page(), 5), 12)
    ok, buf = cv2.imencode(".png", page)
    assert ok
    r = client.post(f"{P}/documents", files=[("files", ("skewed.png", buf.tobytes(), "image/png"))])
    return r.json()["items"][0]["id"]


def test_upload_runs_default_preprocessing(client, doc_id):
    d = client.get(f"{P}/documents/{doc_id}").json()
    assert d["status"] == "ingested"
    page = d["pages"][0]
    assert page["recipe"] == "standard"
    paths = page["stage_paths_json"]
    for key in ("original", "gray", "denoised", "thresholded", "deskewed", "final"):
        assert key in paths
    assert "edges" in paths["extras"]
    assert abs(page["skew_angle"] - 5) <= 0.5
    assert page["quality_before_json"]["sharpness"] > 0 and page["quality_after_json"]["contrast"] > 0


def test_stage_images_are_served(client, doc_id):
    s = client.get(f"{P}/documents/{doc_id}/pages/1/stages").json()
    assert [x["key"] for x in s["stages"]] == ["original", "gray", "denoised", "thresholded", "deskewed", "final"]
    for stage in s["stages"]:
        assert client.get(stage["url"]).status_code == 200
    for url in s["extras"].values():
        assert client.get(url).status_code == 200
    assert s["metrics"]["before"]["skew_deg"] != 0 and len(s["steps"]) == 6
    assert s["steps"][0]["opencv"]


def test_rerun_with_other_recipe_and_compare(client, doc_id):
    r = client.post(
        f"{P}/documents/{doc_id}/preprocess",
        json={"recipe": "basic", "params": {"threshold": {"method": "global", "value": 140}}},
    )
    assert r.status_code == 200
    res = r.json()["pages"][0]
    assert res["recipe"] == "basic" and res["params"]["threshold"]["value"] == 140
    assert {x["key"] for x in res["stages"] if x["skipped"]} == {"deskewed", "final"}

    # active recipe switched, but the earlier run can still be fetched for comparison
    d = client.get(f"{P}/documents/{doc_id}").json()
    assert d["pages"][0]["recipe"] == "basic"
    old = client.get(f"{P}/documents/{doc_id}/pages/1/stages", params={"recipe": "standard"}).json()
    assert old["recipe"] == "standard" and old["active_recipe"] == "basic"


def test_file_route_blocks_path_traversal(client, doc_id, tmp_storage):
    (tmp_storage / "secret.txt").write_text("top secret")
    for evil in ("../secret.txt", "..%2Fsecret.txt", "%2e%2e/secret.txt", "1/../../secret.txt"):
        assert client.get(f"/files/{evil}").status_code == 404
    assert client.get("/files/1").status_code == 404  # a directory, not a file


def test_recipe_catalog(client):
    r = client.get(f"{P}/pipeline/recipes").json()
    names = [x["name"] for x in r["recipes"]]
    assert r["default"] == "standard" and {"baseline", "basic", "gray_enhanced", "standard", "standard_clahe", "full"} <= set(names)
    std = next(x for x in r["recipes"] if x["name"] == "standard")
    assert std["steps"][0]["name"] == "gray" and "cv2" in std["steps"][1]["opencv"]


def test_preprocess_validation_errors(client, doc_id):
    url = f"{P}/documents/{doc_id}/preprocess"
    r = client.post(url, json={"recipe": "magic"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "UNKNOWN_RECIPE"
    r = client.post(url, json={"params": {"threshold": {"method": "bogus"}}})
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_PARAMS"
    r = client.post(url, json={"pages": [99]})
    assert r.status_code == 404
    assert client.post(f"{P}/documents/9999/preprocess", json={}).status_code == 404


def test_stages_errors(client, doc_id):
    assert client.get(f"{P}/documents/{doc_id}/pages/9/stages").status_code == 404
    r = client.get(f"{P}/documents/{doc_id}/pages/1/stages", params={"recipe": "full"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "NOT_PROCESSED"


def test_preprocess_refused_before_pages_exist(client, db):
    from tests.conftest import make_doc

    doc = make_doc(db, status="uploaded", review=None)
    r = client.post(f"{P}/documents/{doc.id}/preprocess", json={})
    assert r.status_code == 409 and r.json()["error"]["code"] == "NOT_READY"

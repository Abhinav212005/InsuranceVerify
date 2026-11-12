import cv2
import numpy as np
import pytest

from app.dip import metrics
from app.dip.pipeline import RECIPES, ParamError, resolve_params, run_pipeline
from app.dip.steps import basic, deskew, geometry, morphology, threshold
from tests.dip_helpers import (
    add_gaussian_noise,
    add_salt_pepper,
    make_page,
    photo_of_page,
    rotate_img,
    uneven_light,
)


@pytest.fixture(scope="module")
def page():
    return make_page()


@pytest.fixture(scope="module")
def gray(page):
    return basic.to_gray(page)


def is_binary(a):
    return set(np.unique(a)).issubset({0, 255})


# ---- 2-4, 14: basic steps -------------------------------------------------------
def test_grayscale_shape_dtype(page):
    g = basic.to_gray(page)
    assert g.ndim == 2 and g.dtype == np.uint8 and g.shape == page.shape[:2]
    assert basic.to_gray(g) is g  # already gray


def test_median_removes_salt_and_pepper(gray):
    noisy = add_salt_pepper(gray, 0.05)
    clean_ref = gray
    before = metrics.psnr_ssim(clean_ref, noisy)["psnr"]
    after = metrics.psnr_ssim(clean_ref, basic.median_blur(noisy, 3))["psnr"]
    assert after > before + 3
    assert metrics.noise_sigma(basic.median_blur(noisy, 3)) < metrics.noise_sigma(noisy)


def test_gaussian_reduces_gaussian_noise(gray):
    noisy = add_gaussian_noise(gray, 25)
    assert metrics.noise_sigma(basic.gaussian_blur(noisy, 5)) < metrics.noise_sigma(noisy)


def test_contrast_enhancement_on_faded_page(gray):
    faded = (gray.astype(np.float32) * 0.25 + 150).astype(np.uint8)  # washed-out print
    base = metrics.contrast(faded)
    # Global equalization stretches the sparse-ink histogram a lot ...
    assert metrics.contrast(basic.hist_equalize(faded)) > base * 3
    # ... while CLAHE is deliberately conservative (clip limit stops it amplifying the huge
    # white-background peak), and gains grow with the clip limit. Worth an experiment.
    c2, c16 = metrics.contrast(basic.clahe(faded, 2)), metrics.contrast(basic.clahe(faded, 16))
    assert c2 > base and c16 > c2


def test_resize_rules():
    small = np.zeros((500, 600), np.uint8)
    out, s = basic.resize_to_width_range(small, 1800, 3000)
    assert out.shape[1] == 1800 and abs(s - 3.0) < 1e-6
    big = np.zeros((4000, 4000), np.uint8)
    out, _ = basic.resize_to_width_range(big, 1800, 3000)
    assert out.shape[1] == 3000
    ok = np.zeros((10, 2480), np.uint8)
    assert basic.resize_to_width_range(ok)[0] is ok


# ---- 5-7: thresholding ------------------------------------------------------------
def test_threshold_outputs_are_binary(gray):
    assert is_binary(threshold.global_threshold(gray))
    out, t = threshold.otsu_threshold(gray)
    assert is_binary(out) and 0 <= t <= 255
    assert is_binary(threshold.adaptive_threshold(gray, 31, 10))


def test_adaptive_beats_global_under_uneven_light(page, gray):
    truth = gray < 128  # text mask of the clean page
    shaded = uneven_light(gray, 0.75)  # right side is bright, left side very dark
    shaded = (shaded.astype(np.float32) * 0.85).astype(np.uint8)

    def accuracy(binary):
        return float(np.mean((binary == 0) == truth))

    g = accuracy(threshold.global_threshold(shaded, 127))
    ad = accuracy(threshold.adaptive_threshold(shaded, 31, 10))
    assert ad > g
    assert ad > 0.9


# ---- 8: deskew ----------------------------------------------------------------------
@pytest.mark.parametrize("angle", [-12, -7, -3, -1, 0, 2, 5, 9, 14])
def test_deskew_angle_error_below_half_degree(gray, angle):
    tilted = rotate_img(gray, angle)
    est = deskew.skew_degrees(tilted)
    assert abs(est - angle) <= 0.5, f"true {angle}, estimated {est}"


def test_deskew_robust_to_shadow_and_noise(gray):
    shadowed = uneven_light(rotate_img(gray, -3), 0.7)
    assert abs(deskew.skew_degrees(shadowed) - (-3)) <= 0.5
    noisy = add_gaussian_noise(rotate_img(gray, 6), 18)
    assert abs(deskew.skew_degrees(noisy) - 6) <= 0.5


def test_adaptive_threshold_auto_c_keeps_noisy_background_clean(gray):
    noisy_boosted = basic.clahe(basic.median_blur(add_gaussian_noise(gray, 18), 3), 2.0)
    ink = float(np.mean(gray < 128))
    fixed = float(np.mean(threshold.adaptive_threshold(noisy_boosted, 31, 10) == 0))
    auto = float(np.mean(threshold.adaptive_threshold(noisy_boosted, 31, 0) == 0))
    assert fixed > ink * 2  # fixed C floods the page with noise specks
    assert auto < ink * 1.5  # auto C stays close to the real ink fraction


def test_deskew_correction_leaves_no_residual_skew(gray):
    tilted = rotate_img(gray, 6)
    fixed = deskew.rotate(tilted, deskew.estimate_correction(tilted))
    assert abs(deskew.skew_degrees(fixed)) <= 0.5


def test_deskew_blank_page_is_noop():
    blank = np.full((600, 400), 255, np.uint8)
    assert deskew.estimate_correction(blank) == 0.0


def test_rotate_keeps_binary_binary(gray):
    b, _ = threshold.otsu_threshold(gray)
    assert is_binary(deskew.rotate(b, 4.0))


# ---- 9: morphology ---------------------------------------------------------------------
def test_morphology_semantics_on_black_text():
    page = np.full((60, 60), 255, np.uint8)
    page[10:20, 10:50] = 0  # a black stroke
    page[40, 40] = 0  # isolated speck
    opened = morphology.apply(page, "open", 3)
    assert opened[40, 40] == 255  # speck removed
    assert opened[12, 20] == 0  # stroke survives

    broken = page.copy()
    broken[14, 28:31] = 255  # small gap in the stroke
    closed = morphology.apply(broken, "close", 5)
    assert closed[14, 29] == 0  # gap reconnected

    assert np.count_nonzero(morphology.apply(page, "dilate", 3) == 0) > np.count_nonzero(page == 0)
    assert np.count_nonzero(morphology.apply(page, "erode", 3) == 0) < np.count_nonzero(page == 0)
    with pytest.raises(ValueError):
        morphology.apply(page, "bogus")


# ---- 10-13: edges, contours, boundary, perspective ----------------------------------------------
def test_canny_and_contours(gray):
    edges = geometry.canny(gray)
    assert is_binary(edges) and edges.any()
    assert len(geometry.find_contours(edges)) > 0


def test_boundary_detection_and_perspective_correction(page):
    photo, true_corners = photo_of_page(page)
    quad = geometry.find_document_quad(photo)
    assert quad is not None
    ordered, truth = geometry.order_points(quad), geometry.order_points(true_corners)
    diag = np.hypot(*photo.shape[:2])
    assert np.max(np.linalg.norm(ordered - truth, axis=1)) < 0.03 * diag

    flat = geometry.four_point_transform(photo, quad)
    # the flattened page has roughly the aspect ratio of the real page (1200x1600)
    assert abs(flat.shape[1] / flat.shape[0] - 1200 / 1600) < 0.12


def test_boundary_returns_none_for_flat_scan(page):
    assert geometry.find_document_quad(page) is None


def test_order_points_is_stable():
    pts = np.array([[100, 100], [0, 0], [100, 0], [0, 100]], np.float32)
    assert geometry.order_points(pts).tolist() == [[0, 0], [100, 0], [100, 100], [0, 100]]


# ---- pipeline -------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def messy(page):
    return add_gaussian_noise(rotate_img(page, 5), 15)


def test_standard_pipeline_end_to_end(messy, tmp_path):
    res = run_pipeline(messy, "standard", None, tmp_path, "/files/x", "/files/x/original.png")
    keys = [s["key"] for s in res["stages"]]
    assert keys == ["original", "gray", "denoised", "thresholded", "deskewed", "final"]
    assert not any(s["skipped"] for s in res["stages"])
    for k in keys[1:]:
        assert (tmp_path / f"{k}.png").exists()
    for extra in ("edges", "contours", "boundary", "cmp_global", "cmp_otsu", "cmp_adaptive", "morph_open", "morph_close"):
        assert (tmp_path / f"{extra}.png").exists()
    assert (tmp_path / "meta.json").exists()
    # skew: found about -5 deg correction, residual is ~0
    assert abs(res["metrics"]["before"]["skew_deg"] - 5) <= 0.5
    assert abs(res["metrics"]["after"]["skew_deg"]) <= 0.5
    assert abs(res["info"]["deskew_correction_deg"] + 5) <= 0.5
    # quality improved
    assert res["metrics"]["after"]["contrast"] > res["metrics"]["before"]["contrast"] * 0.9
    assert res["metrics"]["after"]["noise"] < res["metrics"]["before"]["noise"]
    assert len(res["metrics"]["hist_before"]) == 32 and len(res["steps"]) == 6
    final = cv2.imread(str(tmp_path / "final.png"), cv2.IMREAD_GRAYSCALE)
    assert 1800 <= final.shape[1] <= 3000  # resized into the OCR-friendly width range
    thresholded = cv2.imread(str(tmp_path / "thresholded.png"), cv2.IMREAD_GRAYSCALE)
    assert is_binary(thresholded)


@pytest.mark.parametrize("name", list(RECIPES))
def test_every_recipe_runs(name, messy, tmp_path):
    res = run_pipeline(messy, name, None, tmp_path / name, f"/f/{name}", "/f/o.png", extras=False)
    assert len(res["stages"]) == 6
    assert cv2.imread(str(tmp_path / name / "final.png" if (tmp_path / name / "final.png").exists() else tmp_path / name / "gray.png")) is not None


def test_baseline_marks_missing_stages_skipped(messy, tmp_path):
    res = run_pipeline(messy, "baseline", None, tmp_path, "/f", "/f/o.png", extras=False)
    skipped = {s["key"] for s in res["stages"] if s["skipped"]}
    assert skipped == {"denoised", "thresholded", "deskewed", "final"}


def test_full_recipe_flattens_phone_photo(page, tmp_path):
    photo, _ = photo_of_page(page)
    res = run_pipeline(photo, "full", None, tmp_path, "/f", "/f/o.png", extras=False)
    assert res["info"]["boundary_found"] is True
    # after perspective correction the page is no longer a dark-desk photo: mostly white
    assert cv2.imread(str(tmp_path / "gray.png"), 0).mean() > 200


def test_param_overrides_and_validation():
    r = RECIPES["standard"]
    p = resolve_params(r, {"threshold": {"method": "otsu"}, "denoise": {"ksize": "5"}})
    assert p["threshold"]["method"] == "otsu" and p["denoise"]["ksize"] == 5
    for bad in (
        {"threshold": {"method": "magic"}},
        {"threshold": {"nope": 1}},
        {"boundary": {"min_area": 0.3}},  # step not in this recipe
        {"denoise": {"ksize": "abc"}},
        {"denoise": {"ksize": -3}},
    ):
        with pytest.raises(ParamError):
            resolve_params(r, bad)


def test_preprocessing_leaves_input_untouched(messy, tmp_path):
    snapshot = messy.copy()
    run_pipeline(messy, "standard", None, tmp_path, "/f", "/f/o.png", extras=False)
    assert np.array_equal(snapshot, messy)

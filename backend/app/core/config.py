from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]  # .../backend


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    storage_dir: Path = BASE_DIR / "storage"
    database_url: str = ""
    cors_origins: str = "http://localhost:3000"

    max_upload_mb: int = 20
    max_pages: int = 10
    render_dpi: int = 300
    max_render_side_px: int = 4500  # cap so huge page sizes cannot exhaust memory
    max_image_pixels: int = 60_000_000
    text_layer_min_chars: int = 50  # PDF page counts as "digital" above this

    auto_ocr: bool = True  # run OCR automatically after preprocessing
    default_psm: int = 3
    ocr_low_conf_threshold: float = 60.0  # words below this are flagged for review
    ocr_timeout_s: int = 90

    auto_classify: bool = True  # classify automatically once OCR text exists
    classifier_method: str = "hybrid"  # keyword | ml | hybrid
    auto_extract: bool = True  # extract fields after classification
    auto_validate: bool = True  # validate after extraction
    validation_min_field_confidence: float = 0.6
    validation_min_ocr_confidence: float = 60.0
    validation_min_class_confidence: float = 0.5
    validation_max_amount: str = "1000000000"

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.storage_dir / 'app.db').as_posix()}"


settings = Settings()

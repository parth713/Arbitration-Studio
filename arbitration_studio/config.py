from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    chat_model: str
    embedding_model: str
    # Google Vertex AI (Express Mode) — used for Gemini OCR of scanned MACT docs.
    google_api_key: str
    ocr_model: str
    ocr_dpi: int


def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        chat_model=os.getenv("OPENAI_MODEL", "gpt-4.1").strip(),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
        google_api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
        ocr_model=os.getenv("OCR_MODEL", "gemini-3-pro-preview").strip(),
        ocr_dpi=int(os.getenv("OCR_DPI", "300")),
    )

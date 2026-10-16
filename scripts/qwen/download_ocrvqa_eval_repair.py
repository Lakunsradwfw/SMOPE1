"""Download the seven missing OCR-VQA eval images without re-encoding them."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time

from PIL import Image
import requests


IMAGE_IDS = (
    "1285442962",
    "1421539896",
    "1558210571",
    "1609619919",
    "1931271070",
    "615754090",
    "857194321",
)
OFFICIAL_DATASET_DRIVE_FILE_ID = "1r0tyZUwGCc4wIG4RkiglCGNL_nFJjR6Q"
ARCHIVE_FALLBACKS = {
    # The exact URL in the official metadata now returns 404.  This is a raw
    # replay of its 2014-03-07 HTTP 200 response, not a replacement cover.
    "1421539896": {
        "retrieval_url": (
            "https://web.archive.org/web/20140307193732id_/"
            "http://ecx.images-amazon.com/images/I/611qJzGW%2B9L.jpg"
        ),
        "archive_timestamp": "20140307193732",
        "cdx_digest": "4WDWXMDUAEYIDGDKXOXWANN4SWHKGXE7",
    },
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", required=True,
                        help="dataset.json downloaded from the official OCR-VQA Drive")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_original_bytes(body: bytes) -> tuple[str, int, int]:
    """Read-only validation: never call Image.save or transform downloaded bytes."""
    with Image.open(BytesIO(body)) as image:
        image_format = image.format
        width, height = image.size
        image.verify()
    with Image.open(BytesIO(body)) as image:
        image.convert("RGB").load()
    if image_format not in {"JPEG", "JPG"}:
        raise ValueError(f"Expected JPEG response, got {image_format!r}")
    return image_format, width, height


def download(session: requests.Session, url: str) -> tuple[requests.Response, str]:
    last_error: Exception | None = None
    for attempt in range(4):
        downloaded_at = datetime.now(timezone.utc).isoformat()
        try:
            response = session.get(
                url, timeout=(15, 60), allow_redirects=True,
                headers={"Accept-Encoding": "identity", "User-Agent": "SMOPE-OCRVQA-repair/1.0"})
            if response.status_code == 200:
                return response, downloaded_at
            if response.status_code not in {408, 425, 429, 500, 502, 503, 504}:
                response.raise_for_status()
            last_error = requests.HTTPError(
                f"HTTP {response.status_code} for {url}", response=response)
        except requests.RequestException as error:
            last_error = error
        if attempt < 3:
            time.sleep(2 ** attempt)
    assert last_error is not None
    raise last_error


def main() -> None:
    args = arguments()
    dataset_path = Path(args.dataset_json).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    session = requests.Session()
    # Ignore machine proxy variables: these source URLs are fetched directly,
    # matching the provenance URL recorded by the official dataset metadata.
    session.trust_env = False
    files = []
    for image_id in IMAGE_IDS:
        record = dataset.get(image_id)
        if not isinstance(record, dict):
            raise KeyError(f"Official dataset.json has no image id {image_id}")
        if record.get("split") != 3:
            raise ValueError(f"{image_id} is not in the official test split")
        url = record.get("imageURL") or record.get("imgURL")
        if not isinstance(url, str) or not url.startswith("http"):
            raise ValueError(f"Official dataset record {image_id} has no image URL")
        source_type = "official_dataset_json_imageURL"
        retrieval_url = url
        origin_http_status = None
        archive = None
        try:
            response, downloaded_at = download(session, url)
            origin_http_status = response.status_code
        except requests.HTTPError as error:
            origin_http_status = (
                error.response.status_code if error.response is not None else None
            )
            archive = ARCHIVE_FALLBACKS.get(image_id)
            if origin_http_status != 404 or archive is None:
                raise
            retrieval_url = archive["retrieval_url"]
            response, downloaded_at = download(session, retrieval_url)
            source_type = "internet_archive_raw_capture_of_official_dataset_url"
        body = response.content
        image_format, width, height = validate_original_bytes(body)
        target = output / f"{image_id}.jpg"
        temporary = target.with_suffix(".jpg.part")
        temporary.write_bytes(body)
        temporary.replace(target)
        files.append({
            "image_id": image_id,
            "filename": target.name,
            "url": url,
            "download_utc": downloaded_at,
            "http_status": response.status_code,
            "source_type": source_type,
            "retrieval_url": retrieval_url,
            "origin_http_status_at_download": origin_http_status,
            "final_url": response.url,
            "redirect_history": [item.status_code for item in response.history],
            "size_bytes": len(body),
            "sha256": sha256(body).hexdigest(),
            "format": image_format,
            "width": width,
            "height": height,
            "pillow_verify": True,
            "rgb_decode": True,
            "reencoded": False,
        })
        if archive is not None:
            files[-1]["archive_timestamp"] = archive["archive_timestamp"]
            files[-1]["cdx_digest"] = archive["cdx_digest"]
        print(f"{image_id}: {len(body)} bytes {width}x{height} {files[-1]['sha256']}", flush=True)
    manifest = {
        "format_version": 1,
        "dataset": "OCR-VQA",
        "purpose": "restore seven missing official eval images without re-encoding",
        "official_dataset_metadata": {
            "project_url": "https://ocr-vqa.github.io/",
            "google_drive_file_id": OFFICIAL_DATASET_DRIVE_FILE_ID,
            "local_filename": dataset_path.name,
            "size_bytes": dataset_path.stat().st_size,
            "sha256": file_sha256(dataset_path),
        },
        "files": files,
    }
    manifest_path = output / "eval-image-repair-provenance.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(manifest_path)


if __name__ == "__main__":
    main()

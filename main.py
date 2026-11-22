"""
Payment Screenshot Checker (Telebirr, CBE, CBE Birr, Awash, Dashen, BOA, ...)

What it does
  1. Reads the screenshot with OCR (transaction ID, amount, receiver, provider).
  2. Looks for signs of editing (EXIF software tags, ELA, missing fields).
  3. Detects reused receipts (same image / same transaction ID) with SQLite.
  4. Confirms the transaction with the provider's official receipt page (Telebirr
     is built in; other banks can be added in PROVIDERS).

IMPORTANT: An image alone can NEVER prove a payment is real. Only step 4
(asking the provider) is proof. Steps 1-3 only catch lazy fakes and reuse.

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000
Docs:  http://localhost:8000/docs
"""
import hashlib
import io
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import httpx
import imagehash
import numpy as np
import pytesseract
from bs4 import BeautifulSoup
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from PIL import ExifTags, Image, ImageChops

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
API_KEY = os.getenv("API_KEY", "")  # leave empty to disable auth (dev only)
DB_PATH = os.getenv("DB_PATH", "receipts.db")
VERIFY_SSL = os.getenv("VERIFY_SSL", "true").lower() == "true"
OCR_LANG = os.getenv("OCR_LANG", "eng")  # use "eng+amh" if amh traineddata installed
MAX_IMAGE_BYTES = 8 * 1024 * 1024

# Receipt lookup URLs. {txn} is replaced with the transaction ID.
# Telebirr's public receipt page - confirm the URL still works for you.
# Banks that need an account suffix or an API key must be added by you.
PROVIDERS = {
    "telebirr": {
        "keywords": ["telebirr", "ethio telecom", "ethiotelecom", "ethio-telecom"],
        "receipt_url": os.getenv(
            "TELEBIRR_RECEIPT_URL",
            "https://transactioninfo.ethiotelecom.et/receipt/{txn}",
        ),
    },
    "cbe": {
        "keywords": ["commercial bank of ethiopia", "cbe birr", "cbebirr", "cbe"],
        "receipt_url": os.getenv("CBE_RECEIPT_URL", ""),  # needs account suffix
    },
    "awash": {"keywords": ["awash bank", "awash"], "receipt_url": ""},
    "dashen": {"keywords": ["dashen bank", "dashen", "amole"], "receipt_url": ""},
    "boa": {"keywords": ["bank of abyssinia", "abyssinia", "boa"], "receipt_url": ""},
    "mpesa": {"keywords": ["m-pesa", "mpesa", "safaricom"], "receipt_url": ""},
}

EDITOR_TAGS = [
    "photoshop", "gimp", "canva", "snapseed", "picsart", "pixlr", "lightroom",
    "paint.net", "affinity", "photopea", "fotor", "inpaint", "meitu", "facetune",
]

app = FastAPI(title="Payment Screenshot Checker", version="1.0")


# ----------------------------------------------------------------------------
# Auth + DB
# ----------------------------------------------------------------------------
def auth(x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            txn TEXT, sha256 TEXT, phash TEXT, provider TEXT,
            amount REAL, status TEXT, created_at TEXT)"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS i_txn ON receipts(txn)")
    conn.execute("CREATE INDEX IF NOT EXISTS i_sha ON receipts(sha256)")
    return conn


# ----------------------------------------------------------------------------
# OCR + field extraction
# ----------------------------------------------------------------------------
def run_ocr(img: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(img, lang=OCR_LANG)
    except pytesseract.TesseractError:
        return pytesseract.image_to_string(img, lang="eng")


def detect_provider(text: str) -> Optional[str]:
    low = text.lower()
    # check longer / more specific keywords first
    for name, cfg in PROVIDERS.items():
        for kw in sorted(cfg["keywords"], key=len, reverse=True):
            if kw in low:
                return name
    return None


def extract_txn(text: str) -> Optional[str]:
    # 1) token right after a "transaction" label
    m = re.search(
        r"(?:transaction\s*(?:no|number|id)\.?|receipt\s*no\.?|ref(?:erence)?\s*(?:no|number)?\.?)"
        r"\s*[:\-]?\s*([A-Z0-9]{8,20})",
        text,
        re.I,
    )
    if m:
        return m.group(1).upper()
    # 2) fallback: any 10-char token with letters AND digits (Telebirr style)
    for tok in re.findall(r"\b[A-Z0-9]{10}\b", text.upper()):
        if re.search(r"\d", tok) and re.search(r"[A-Z]", tok):
            return tok
    return None


def extract_amount(text: str) -> Optional[float]:
    patterns = [
        r"(?:total\s*paid\s*amount|amount|total)\s*[:\-]?\s*(?:ETB|Birr)?\s*([\d,]+(?:\.\d{1,2})?)",
        r"([\d,]+(?:\.\d{1,2})?)\s*(?:ETB|Birr)",
    ]
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                pass
    return None


def extract_receiver(text: str) -> Optional[str]:
    m = re.search(
        r"(?:receiver|credited\s*party|paid\s*to|to)\s*(?:name)?\s*[:\-]\s*([A-Za-z .'-]{3,60})",
        text,
        re.I,
    )
    return m.group(1).strip() if m else None


def extract_date(text: str) -> Optional[str]:
    m = re.search(r"(\d{1,4}[-/]\d{1,2}[-/]\d{1,4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?)", text)
    return m.group(1) if m else None


# ----------------------------------------------------------------------------
# Image forensics (weak signals, never proof)
# ----------------------------------------------------------------------------
def exif_flags(img: Image.Image) -> list[str]:
    flags = []
    try:
        exif = img.getexif()
        for tag_id, value in exif.items():
            name = ExifTags.TAGS.get(tag_id, str(tag_id))
            if name in ("Software", "ProcessingSoftware", "Artist"):
                if any(e in str(value).lower() for e in EDITOR_TAGS):
                    flags.append(f"Edited with image editor: {value}")
    except Exception:
        pass
    info_sw = str(img.info.get("Software", "")).lower()
    if any(e in info_sw for e in EDITOR_TAGS):
        flags.append(f"Edited with image editor: {info_sw}")
    return flags


def ela_ratio(img: Image.Image, quality: int = 90) -> float:
    """Error Level Analysis: pasted/edited regions often compress differently.
    Returns max-block / median-block error ratio. Unreliable for PNG screenshots."""
    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    diff = ImageChops.difference(rgb, Image.open(buf))
    arr = np.asarray(diff).astype(float).sum(axis=2)
    h, w = arr.shape
    b = 32
    blocks = [
        arr[y : y + b, x : x + b].mean()
        for y in range(0, h - b + 1, b)
        for x in range(0, w - b + 1, b)
    ]
    if not blocks:
        return 0.0
    return float(np.max(blocks) / (np.median(blocks) + 1e-6))


# ----------------------------------------------------------------------------
# Provider confirmation (the only real proof)
# ----------------------------------------------------------------------------
async def confirm_with_provider(
    provider: Optional[str], txn: str, amount: Optional[float], receiver: Optional[str]
) -> dict:
    cfg = PROVIDERS.get(provider or "")
    if not cfg or not cfg["receipt_url"]:
        return {"checked": False, "reason": f"No receipt lookup configured for '{provider}'"}

    url = cfg["receipt_url"].format(txn=txn)
    try:
        async with httpx.AsyncClient(timeout=15, verify=VERIFY_SSL, follow_redirects=True) as c:
            r = await c.get(url, headers={"User-Agent": "Mozilla/5.0"})
    except httpx.HTTPError as e:
        return {"checked": False, "reason": f"Could not reach provider: {e.__class__.__name__}"}

    if r.status_code == 404:
        return {"checked": True, "found": False, "reason": "Provider has no such transaction"}
    if r.status_code != 200:
        return {"checked": False, "reason": f"Provider returned HTTP {r.status_code}"}

    text = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
    flat = text.replace(",", "")
    found = txn.lower() in text.lower()
    if not found:
        return {"checked": True, "found": False, "reason": "Receipt page does not contain this transaction ID"}

    result = {"checked": True, "found": True, "amount_match": None, "receiver_match": None}
    if amount is not None:
        result["amount_match"] = (f"{amount:.2f}" in flat) or (f"{amount:.0f}" in re.findall(r"\d+", flat))
    if receiver:
        words = [w for w in re.split(r"\s+", receiver.lower()) if len(w) > 2]
        result["receiver_match"] = bool(words) and all(w in text.lower() for w in words)
    return result


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def decide(score: int, provider_result: dict, hard_fake: bool) -> str:
    if hard_fake:
        return "FAKE"
    if provider_result.get("checked") and provider_result.get("found"):
        if provider_result.get("amount_match") is False or provider_result.get("receiver_match") is False:
            return "FAKE"
        return "REAL" if score >= 60 else "SUSPICIOUS"
    if provider_result.get("checked") and provider_result.get("found") is False:
        return "FAKE"
    # could not confirm with provider
    return "UNVERIFIED" if score >= 60 else "SUSPICIOUS"


async def analyze(
    img: Image.Image,
    raw: bytes,
    expected_amount: Optional[float],
    expected_receiver: Optional[str],
    register: bool,
) -> dict:
    score, reasons, hard_fake = 100, [], False

    text = run_ocr(img)
    provider = detect_provider(text)
    txn = extract_txn(text)
    amount = extract_amount(text)
    receiver = extract_receiver(text)
    date = extract_date(text)

    if not provider:
        score -= 20
        reasons.append("Could not identify the payment provider from the image")
    if not txn:
        score -= 30
        reasons.append("No transaction ID found in the image")
    if amount is None:
        score -= 15
        reasons.append("No amount found in the image")

    # expectations from your order
    if expected_amount is not None and amount is not None and abs(amount - expected_amount) > 0.01:
        hard_fake = True
        reasons.append(f"Amount on image ({amount}) differs from expected ({expected_amount})")
    if expected_receiver and receiver:
        if not all(w in receiver.lower() for w in expected_receiver.lower().split() if len(w) > 2):
            score -= 25
            reasons.append("Receiver name on image does not match expected receiver")

    # forensics
    for f in exif_flags(img):
        score -= 35
        reasons.append(f)
    fmt = (img.format or "").upper()
    if fmt in ("JPEG", "JPG"):
        ratio = ela_ratio(img)
        if ratio > 8:
            score -= 15
            reasons.append(f"Uneven compression (ELA ratio {ratio:.1f}) - possible edited region")

    # duplicates
    sha = hashlib.sha256(raw).hexdigest()
    ph = str(imagehash.phash(img))
    conn = db()
    dup_img = conn.execute("SELECT 1 FROM receipts WHERE sha256=?", (sha,)).fetchone()
    dup_txn = conn.execute("SELECT 1 FROM receipts WHERE txn=? AND status IN ('REAL','UNVERIFIED')", (txn,)).fetchone() if txn else None
    near_dup = False
    for (old,) in conn.execute("SELECT phash FROM receipts WHERE phash IS NOT NULL ORDER BY id DESC LIMIT 5000"):
        try:
            if imagehash.hex_to_hash(old) - imagehash.phash(img) <= 4 and old != ph:
                near_dup = True
                break
        except Exception:
            continue
    if dup_img or dup_txn:
        hard_fake = True
        reasons.append("This receipt/transaction ID was already submitted before (reuse)")
    elif near_dup:
        score -= 25
        reasons.append("Image is very similar to a previous receipt but not identical (possible edited copy)")

    # provider confirmation
    provider_result = {"checked": False, "reason": "No transaction ID"}
    if txn:
        provider_result = await confirm_with_provider(provider, txn, amount, receiver)
        if provider_result.get("checked") and provider_result.get("found") is False:
            reasons.append(provider_result["reason"])
        elif provider_result.get("found"):
            if provider_result.get("amount_match") is False:
                reasons.append("Amount on image does not match the provider's record")
            if provider_result.get("receiver_match") is False:
                reasons.append("Receiver on image does not match the provider's record")
        else:
            reasons.append(f"Not confirmed with provider: {provider_result.get('reason')}")
    verdict = decide(max(score, 0), provider_result, hard_fake)

    if register:
        conn.execute(
            "INSERT INTO receipts (txn, sha256, phash, provider, amount, status, created_at) VALUES (?,?,?,?,?,?,?)",
            (txn, sha, ph, provider, amount, verdict, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    conn.close()

    return {
        "verdict": verdict,  # REAL | FAKE | SUSPICIOUS | UNVERIFIED
        "score": max(score, 0),
        "reasons": reasons,
        "extracted": {
            "provider": provider,
            "transaction_id": txn,
            "amount": amount,
            "receiver": receiver,
            "date": date,
        },
        "provider_check": provider_result,
        "note": "REAL is only returned when the provider's own records confirm the transaction.",
    }


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok", "docs": "/docs"}


@app.post("/verify-image", dependencies=[Depends(auth)])
async def verify_image(
    file: UploadFile = File(...),
    expected_amount: Optional[float] = Form(None),
    expected_receiver: Optional[str] = Form(None),
    register: bool = Form(True),
):
    raw = await file.read()
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large (max 8 MB)")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        raise HTTPException(400, "File is not a valid image")
    return await analyze(img, raw, expected_amount, expected_receiver, register)


@app.post("/verify-transaction", dependencies=[Depends(auth)])
async def verify_transaction(
    transaction_id: str = Form(...),
    provider: str = Form("telebirr"),
    expected_amount: Optional[float] = Form(None),
    expected_receiver: Optional[str] = Form(None),
):
    """Check a transaction ID directly (no image). Most reliable method."""
    res = await confirm_with_provider(provider.lower(), transaction_id.strip().upper(), expected_amount, expected_receiver)
    if res.get("checked") and res.get("found"):
        ok = res.get("amount_match") is not False and res.get("receiver_match") is not False
        verdict = "REAL" if ok else "FAKE"
    elif res.get("checked"):
        verdict = "FAKE"
    else:
        verdict = "UNVERIFIED"
    return {"verdict": verdict, "provider_check": res}

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
    uvicorn main:app --host 0.0.0.0 --port 8000
Docs:  http://localhost:8000/docs
UI:    http://localhost:8000/ui
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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import ExifTags, Image, ImageChops

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
API_KEY = os.getenv("API_KEY", "")  # leave empty to disable auth (dev only)
DB_PATH = os.getenv("DB_PATH", "/tmp/receipts.db")
VERIFY_SSL = os.getenv("VERIFY_SSL", "true").lower() == "true"
OCR_LANG = os.getenv("OCR_LANG", "eng")  # use "eng+amh" if amh traineddata installed
MAX_IMAGE_BYTES = 8 * 1024 * 1024

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------
# Auth + DB
# ----------------------------------------------------------------------------
def auth(x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def db():
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
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
    for name, cfg in PROVIDERS.items():
        for kw in sorted(cfg["keywords"], key=len, reverse=True):
            if kw in low:
                return name
    return None


def extract_txn(text: str) -> Optional[str]:
    m = re.search(
        r"(?:transaction\s*(?:no|number|id)\.?|receipt\s*no\.?|ref(?:erence)?\s*(?:no|number)?\.?)"
        r"\s*[:\-]?\s*([A-Z0-9]{8,20})",
        text,
        re.I,
    )
    if m:
        return m.group(1).upper()
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
        r"(?:receiver|credited\s*party|paid\s*to|\bto)\s*(?:name)?\s*[:\-]\s*([A-Za-z .'-]{3,60})",
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
# Provider confirmation
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

    if expected_amount is not None and amount is not None and abs(amount - expected_amount) > 0.01:
        hard_fake = True
        reasons.append(f"Amount on image ({amount}) differs from expected ({expected_amount})")
    if expected_receiver and receiver:
        if not all(w in receiver.lower() for w in expected_receiver.lower().split() if len(w) > 2):
            score -= 25
            reasons.append("Receiver name on image does not match expected receiver")

    for f in exif_flags(img):
        score -= 35
        reasons.append(f)
    fmt = (img.format or "").upper()
    if fmt in ("JPEG", "JPG"):
        ratio = ela_ratio(img)
        if ratio > 8:
            score -= 15
            reasons.append(f"Uneven compression (ELA ratio {ratio:.1f}) - possible edited region")

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
        "verdict": verdict,
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
# HTML + CSS UI
# ----------------------------------------------------------------------------
UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Payment Screenshot Checker</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-hover: #0ea5e9;
      --ok: #22c55e;
      --warn: #f59e0b;
      --bad: #ef4444;
    }
    * { box-sizing: border-box; font-family: system-ui, -apple-system, sans-serif; }
    body {
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    .container { max-width: 820px; margin: 0 auto; }
    h1 { margin: 0 0 6px; font-size: 1.55rem; }
    .sub { margin: 0 0 20px; color: var(--muted); font-size: 0.92rem; }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 18px;
    }
    .grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }
    @media (max-width: 640px) { .grid-2 { grid-template-columns: 1fr; } }
    .field { margin-bottom: 14px; }
    label { display: block; font-size: 0.85rem; color: var(--muted); margin-bottom: 6px; }
    input, select {
      width: 100%;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--bg);
      color: var(--text);
      font-size: 0.95rem;
    }
    input:focus, select:focus {
      outline: 2px solid var(--accent);
      border-color: var(--accent);
    }
    .checkbox-label {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--text);
      font-size: 0.9rem;
      margin-bottom: 16px;
      cursor: pointer;
    }
    .checkbox-label input[type="checkbox"] { width: 16px; height: 16px; margin: 0; accent-color: var(--accent); }
    .tabs { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
    .btn {
      padding: 9px 14px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: transparent;
      color: var(--text);
      font-size: 0.9rem;
      cursor: pointer;
    }
    .btn:hover { border-color: var(--accent); }
    .btn.active {
      background: var(--accent);
      color: #0f172a;
      border-color: var(--accent);
      font-weight: 600;
    }
    .btn-primary {
      background: var(--accent);
      color: #0f172a;
      border: none;
      padding: 11px 18px;
      border-radius: 8px;
      font-size: 0.95rem;
      font-weight: 700;
      cursor: pointer;
    }
    .btn-primary:hover { background: var(--accent-hover); }
    .action-row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
    .status-text { font-size: 0.9rem; color: var(--muted); }
    .badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 700;
      margin-bottom: 10px;
    }
    .REAL { background: rgba(34, 197, 94, 0.2); color: var(--ok); }
    .UNVERIFIED, .SUSPICIOUS { background: rgba(245, 158, 11, 0.2); color: var(--warn); }
    .FAKE { background: rgba(239, 68, 68, 0.2); color: var(--bad); }
    pre {
      margin: 0;
      padding: 14px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #020617;
      color: #e2e8f0;
      font-size: 0.85rem;
      overflow-x: auto;
    }
  </style>
</head>
<body>
  <main class="container">
    <h1>Payment Screenshot Checker</h1>
    <p class="sub">Verify payment screenshots or transaction IDs on Render.</p>

    <section class="card">
      <div class="field">
        <label for="apiKey">API Key (x-api-key, optional if API_KEY is not set)</label>
        <input id="apiKey" type="password" placeholder="Enter API_KEY if configured" />
      </div>
      <div class="action-row">
        <button id="healthBtn" class="btn" type="button">Check Health (/)</button>
        <span id="healthStatus" class="status-text">Ready</span>
      </div>
    </section>

    <section class="card">
      <div class="tabs">
        <button id="tabImg" class="btn active" type="button">Verify Screenshot</button>
        <button id="tabTxn" class="btn" type="button">Verify Transaction ID</button>
      </div>

      <form id="imgForm">
        <div class="field">
          <label for="file">Receipt Screenshot (PNG, JPG, max 8 MB)</label>
          <input id="file" type="file" accept="image/*" required />
        </div>
        <div class="grid-2">
          <div class="field">
            <label for="imgAmount">Expected Amount (optional)</label>
            <input id="imgAmount" type="number" step="0.01" placeholder="e.g. 500.00" />
          </div>
          <div class="field">
            <label for="imgReceiver">Expected Receiver (optional)</label>
            <input id="imgReceiver" type="text" placeholder="e.g. Receiver Name" />
          </div>
        </div>
        <label class="checkbox-label">
          <input id="register" type="checkbox" checked />
          Save receipt to database to detect reuse
        </label>
        <button class="btn-primary" type="submit">Analyze Screenshot</button>
      </form>

      <form id="txnForm" style="display:none;">
        <div class="grid-2">
          <div class="field">
            <label for="txnId">Transaction ID</label>
            <input id="txnId" type="text" required placeholder="e.g. TX1234567890" />
          </div>
          <div class="field">
            <label for="provider">Provider</label>
            <select id="provider">
              <option value="telebirr">telebirr</option>
              <option value="cbe">cbe</option>
              <option value="awash">awash</option>
              <option value="dashen">dashen</option>
              <option value="boa">boa</option>
              <option value="mpesa">mpesa</option>
            </select>
          </div>
        </div>
        <div class="grid-2">
          <div class="field">
            <label for="txnAmount">Expected Amount (optional)</label>
            <input id="txnAmount" type="number" step="0.01" placeholder="e.g. 500.00" />
          </div>
          <div class="field">
            <label for="txnReceiver">Expected Receiver (optional)</label>
            <input id="txnReceiver" type="text" placeholder="e.g. Receiver Name" />
          </div>
        </div>
        <button class="btn-primary" type="submit">Verify Transaction</button>
      </form>
    </section>

    <section class="card">
      <div id="verdictBadge"></div>
      <pre id="output">Results will appear here...</pre>
    </section>
  </main>

  <script>
    const outputEl = document.getElementById("output");
    const badgeEl = document.getElementById("verdictBadge");

    function getHeaders() {
      const k = document.getElementById("apiKey").value.trim();
      return k ? { "x-api-key": k } : {};
    }

    function showResult(data) {
      if (data && data.verdict) {
        const v = String(data.verdict).toUpperCase();
        badgeEl.innerHTML = `<span class="badge ${v}">${v}</span>`;
      } else {
        badgeEl.innerHTML = "";
      }
      outputEl.textContent = JSON.stringify(data, null, 2);
    }

    document.getElementById("tabImg").addEventListener("click", () => {
      document.getElementById("imgForm").style.display = "block";
      document.getElementById("txnForm").style.display = "none";
      document.getElementById("tabImg").classList.add("active");
      document.getElementById("tabTxn").classList.remove("active");
    });

    document.getElementById("tabTxn").addEventListener("click", () => {
      document.getElementById("imgForm").style.display = "none";
      document.getElementById("txnForm").style.display = "block";
      document.getElementById("tabTxn").classList.add("active");
      document.getElementById("tabImg").classList.remove("active");
    });

    document.getElementById("healthBtn").addEventListener("click", async () => {
      const st = document.getElementById("healthStatus");
      st.textContent = "Checking...";
      try {
        const r = await fetch("/", { headers: getHeaders() });
        const d = await r.json();
        st.textContent = r.ok ? `Online (${r.status})` : `Error (${r.status})`;
        showResult(d);
      } catch (err) {
        st.textContent = "Offline / Error";
        showResult({ error: err.message });
      }
    });

    document.getElementById("imgForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      const f = document.getElementById("file").files[0];
      if (!f) return;
      const fd = new FormData();
      fd.append("file", f);
      const amt = document.getElementById("imgAmount").value;
      const rcv = document.getElementById("imgReceiver").value.trim();
      if (amt !== "") fd.append("expected_amount", amt);
      if (rcv !== "") fd.append("expected_receiver", rcv);
      fd.append("register", document.getElementById("register").checked);
      badgeEl.innerHTML = "";
      outputEl.textContent = "Analyzing screenshot...";
      try {
        const r = await fetch("/verify-image", { method: "POST", headers: getHeaders(), body: fd });
        showResult(await r.json());
      } catch (err) {
        showResult({ error: err.message });
      }
    });

    document.getElementById("txnForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      const fd = new FormData();
      fd.append("transaction_id", document.getElementById("txnId").value.trim());
      fd.append("provider", document.getElementById("provider").value);
      const amt = document.getElementById("txnAmount").value;
      const rcv = document.getElementById("txnReceiver").value.trim();
      if (amt !== "") fd.append("expected_amount", amt);
      if (rcv !== "") fd.append("expected_receiver", rcv);
      badgeEl.innerHTML = "";
      outputEl.textContent = "Verifying transaction...";
      try {
        const r = await fetch("/verify-transaction", { method: "POST", headers: getHeaders(), body: fd });
        showResult(await r.json());
      } catch (err) {
        showResult({ error: err.message });
      }
    });
  </script>
</body>
</html>"""


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok", "docs": "/docs", "ui": "/ui"}


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return HTMLResponse(content=UI_HTML)


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

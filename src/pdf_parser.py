"""
Loan Report PDF Parser
=======================
Extracts past loan information from uploaded PDF documents
using text extraction and regex pattern matching.

Handles various PDF formats and extracts:
- Loan amounts
- Loan statuses (Repaid/Closed/Defaulted/Active)
- Loan types
- Dates
- Lender names

Author: Kinshunk Garg
"""

import re
import io
import pandas as pd

try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False


# Status keyword mappings
STATUS_KEYWORDS = {
    "Repaid": [
        "repaid", "closed", "paid off", "settled", "fully paid",
        "completed", "cleared", "discharged", "paid in full",
        "closure", "loan closed", "account closed", "prepaid"
    ],
    "Defaulted": [
        "default", "defaulted", "npa", "non-performing", "write-off",
        "written off", "overdue", "delinquent", "bad debt", "loss",
        "doubtful", "sub-standard", "substandard", "90+ days past due",
        "charge off", "charged off"
    ],
    "Active": [
        "active", "ongoing", "current", "open", "in progress",
        "emi running", "disbursed", "sanctioned", "regular",
        "standard", "performing"
    ],
}

# Loan type keywords
LOAN_TYPE_KEYWORDS = {
    "Home Loan": ["home loan", "housing loan", "mortgage", "home finance", "housing finance"],
    "Personal Loan": ["personal loan", "consumer loan", "unsecured loan"],
    "Car Loan": ["car loan", "auto loan", "vehicle loan", "automobile loan"],
    "Education Loan": ["education loan", "student loan", "study loan", "edu loan"],
    "Business Loan": ["business loan", "commercial loan", "msme loan", "enterprise loan"],
    "Gold Loan": ["gold loan", "jewel loan"],
    "Two-Wheeler Loan": ["two-wheeler", "two wheeler", "bike loan", "scooter loan"],
    "Credit Card": ["credit card", "cc outstanding", "revolving credit"],
    "Consumer Durable Loan": ["consumer durable", "appliance loan", "electronics loan"],
}

# Amount extraction patterns
AMOUNT_PATTERNS = [
    # Rs./INR/₹ followed by a number (with optional commas and decimals)
    r'(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d{1,2})?)',
    # $ followed by a number
    r'\$\s*([\d,]+(?:\.\d{1,2})?)',
    # "Amount: 1,50,000" or "Loan Amount: 15,00,000"
    r'(?:loan\s*)?amount\s*[:\-]?\s*(?:Rs\.?|INR|₹|\$)?\s*([\d,]+(?:\.\d{1,2})?)',
    # "1,50,000 INR" or "150000 Rs"
    r'([\d,]+(?:\.\d{1,2})?)\s*(?:INR|Rs\.?|rupees)',
    # Standalone large numbers (likely loan amounts) — 5+ digits
    r'\b(\d{1,2},?\d{2},?\d{3}(?:\.\d{1,2})?)\b',
]

# Date patterns
DATE_PATTERNS = [
    r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',          # 2021-03-15
    r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',          # 15-03-2021
    r'(\d{1,2}[-/]\d{1,2}[-/]\d{2})\b',       # 15-03-21
    r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',  # 15 March 2021
    r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})',  # March 15, 2021
]


def parse_amount(amount_str: str) -> float:
    """Parse an amount string into a float value in $1000s."""
    # Remove commas and spaces
    cleaned = amount_str.replace(",", "").replace(" ", "").strip()
    try:
        val = float(cleaned)
        # If value is large (likely in rupees/dollars), convert to $1000s
        if val > 1000:
            return round(val / 1000, 1)
        return round(val, 1)
    except ValueError:
        return 0.0


def detect_status(text_chunk: str) -> str:
    """Detect loan status from a text chunk."""
    text_lower = text_chunk.lower()
    for status, keywords in STATUS_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return status
    return "Unknown"


def detect_loan_type(text_chunk: str) -> str:
    """Detect loan type from a text chunk."""
    text_lower = text_chunk.lower()
    for loan_type, keywords in LOAN_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return loan_type
    return "Unspecified Loan"


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    if not HAS_PYPDF2:
        raise ImportError(
            "PyPDF2 is required for PDF parsing. Install it with: pip install PyPDF2"
        )

    reader = PdfReader(io.BytesIO(file_bytes))
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    return "\n".join(text_parts)


def parse_loan_pdf(file_bytes: bytes) -> dict:
    """
    Parse a loan report PDF and extract structured loan data.

    Args:
        file_bytes: Raw bytes of the PDF file

    Returns:
        dict with:
            - success (bool)
            - raw_text (str)
            - loans_found (int)
            - prev_loan_count (int) — completed loans
            - prev_loans_repaid (int)
            - prev_loan_defaults (int)
            - avg_prev_loan_amount (float)
            - repayment_rate (float)
            - records (list of dicts)
            - error (str or None)
    """
    try:
        text = extract_text_from_pdf(file_bytes)
    except Exception as e:
        return {
            "success": False,
            "raw_text": "",
            "loans_found": 0,
            "prev_loan_count": 0,
            "prev_loans_repaid": 0,
            "prev_loan_defaults": 0,
            "avg_prev_loan_amount": 0,
            "repayment_rate": 0.0,
            "records": [],
            "error": f"Failed to read PDF: {str(e)}",
        }

    if not text.strip():
        return {
            "success": False,
            "raw_text": "",
            "loans_found": 0,
            "prev_loan_count": 0,
            "prev_loans_repaid": 0,
            "prev_loan_defaults": 0,
            "avg_prev_loan_amount": 0,
            "repayment_rate": 0.0,
            "records": [],
            "error": "Could not extract any text from the PDF. The file may be scanned/image-based.",
        }

    # --- Strategy 1: Line-by-line parsing ---
    records = []
    lines = text.split("\n")

    # Try to detect table-like structures or loan entries
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Look for lines that contain both an amount and a status keyword
        status = detect_status(line_stripped)
        loan_type = detect_loan_type(line_stripped)

        # Extract amounts from this line
        amounts = []
        for pattern in AMOUNT_PATTERNS[:4]:  # Skip standalone number pattern
            matches = re.findall(pattern, line_stripped, re.IGNORECASE)
            for m in matches:
                amt = parse_amount(m)
                if amt >= 1:  # At least $1000
                    amounts.append(amt)

        # Extract dates
        dates = []
        for pattern in DATE_PATTERNS:
            matches = re.findall(pattern, line_stripped, re.IGNORECASE)
            dates.extend(matches)

        # If we found a status and at least something else interesting
        if status != "Unknown" and (amounts or loan_type != "Unspecified Loan"):
            record = {
                "Loan_Type": loan_type,
                "Loan_Amount": amounts[0] if amounts else 0,
                "Loan_Date": dates[0] if dates else "N/A",
                "Status": status,
            }
            records.append(record)
        elif amounts and loan_type != "Unspecified Loan":
            # We found a loan type and amount but no clear status
            # Look at surrounding lines for status
            context = " ".join(lines[max(0, i - 2): min(len(lines), i + 3)])
            ctx_status = detect_status(context)
            record = {
                "Loan_Type": loan_type,
                "Loan_Amount": amounts[0],
                "Loan_Date": dates[0] if dates else "N/A",
                "Status": ctx_status if ctx_status != "Unknown" else "Active",
            }
            records.append(record)

    # --- Strategy 2: Block-based parsing (if line-by-line found nothing) ---
    if not records:
        # Split text into paragraphs/blocks
        blocks = re.split(r'\n\s*\n|\n(?=\d+[.)]\s)', text)
        for block in blocks:
            if len(block.strip()) < 10:
                continue

            status = detect_status(block)
            loan_type = detect_loan_type(block)

            amounts = []
            for pattern in AMOUNT_PATTERNS[:4]:
                matches = re.findall(pattern, block, re.IGNORECASE)
                for m in matches:
                    amt = parse_amount(m)
                    if amt >= 1:
                        amounts.append(amt)

            dates = []
            for pattern in DATE_PATTERNS:
                matches = re.findall(pattern, block, re.IGNORECASE)
                dates.extend(matches)

            if (status != "Unknown" or loan_type != "Unspecified Loan") and amounts:
                record = {
                    "Loan_Type": loan_type,
                    "Loan_Amount": amounts[0],
                    "Loan_Date": dates[0] if dates else "N/A",
                    "Status": status if status != "Unknown" else "Active",
                }
                records.append(record)

    # --- Strategy 3: Global extraction (last resort) ---
    if not records:
        # Count status keywords globally
        text_lower = text.lower()
        repaid_count = sum(
            text_lower.count(kw) for kw in STATUS_KEYWORDS["Repaid"]
        )
        default_count = sum(
            text_lower.count(kw) for kw in STATUS_KEYWORDS["Defaulted"]
        )
        active_count = sum(
            text_lower.count(kw) for kw in STATUS_KEYWORDS["Active"]
        )

        # Extract all amounts
        all_amounts = []
        for pattern in AMOUNT_PATTERNS[:4]:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                amt = parse_amount(m)
                if amt >= 1:
                    all_amounts.append(amt)

        # Build synthetic records from global counts
        total = max(repaid_count + default_count + active_count, len(all_amounts))
        if total > 0:
            for j in range(min(total, 8)):  # Cap at 8
                if j < repaid_count:
                    st = "Repaid"
                elif j < repaid_count + default_count:
                    st = "Defaulted"
                else:
                    st = "Active"

                amt = all_amounts[j] if j < len(all_amounts) else 50
                records.append({
                    "Loan_Type": "Unspecified Loan",
                    "Loan_Amount": amt,
                    "Loan_Date": "N/A",
                    "Status": st,
                })

    # --- Compute summary stats ---
    completed = [r for r in records if r["Status"] in ("Repaid", "Defaulted")]
    repaid = [r for r in completed if r["Status"] == "Repaid"]
    defaulted = [r for r in completed if r["Status"] == "Defaulted"]

    all_amounts = [r["Loan_Amount"] for r in records if r["Loan_Amount"] > 0]
    avg_amount = round(sum(all_amounts) / len(all_amounts), 1) if all_amounts else 0

    n_completed = len(completed)
    n_repaid = len(repaid)
    n_defaulted = len(defaulted)
    repayment_rate = round(n_repaid / n_completed, 2) if n_completed > 0 else 0.0

    return {
        "success": True,
        "raw_text": text[:2000],  # First 2000 chars for preview
        "loans_found": len(records),
        "prev_loan_count": n_completed,
        "prev_loans_repaid": n_repaid,
        "prev_loan_defaults": n_defaulted,
        "avg_prev_loan_amount": avg_amount,
        "repayment_rate": repayment_rate,
        "records": records,
        "error": None,
    }

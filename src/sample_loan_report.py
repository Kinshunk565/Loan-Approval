"""
Sample Loan Report PDF Generator
==================================
Generates a realistic sample loan report PDF that users can
download as a template or use for testing the PDF upload feature.

Uses reportlab if available, otherwise falls back to a simple
text-based PDF using PyPDF2 (write mode).

Author: Kinshunk Garg
"""

import os
import io
import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    from PyPDF2 import PdfWriter
    from PyPDF2.generic import AnnotationBuilder
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False


def generate_sample_pdf_reportlab(person_name: str = "Rahul Sharma",
                                   person_id: str = "PER00001") -> bytes:
    """Generate a professional sample loan report PDF using reportlab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'],
        fontSize=20, spaceAfter=6, alignment=TA_CENTER,
        textColor=colors.HexColor('#A91D22')
    )
    subtitle_style = ParagraphStyle(
        'CustomSubtitle', parent=styles['Normal'],
        fontSize=10, spaceAfter=20, alignment=TA_CENTER,
        textColor=colors.grey
    )
    header_style = ParagraphStyle(
        'SectionHeader', parent=styles['Heading2'],
        fontSize=13, spaceAfter=8, spaceBefore=16,
        textColor=colors.HexColor('#A91D22'),
        borderPadding=(0, 0, 4, 0)
    )
    normal_style = styles['Normal']

    elements = []

    # Title
    elements.append(Paragraph("LOAN HISTORY REPORT", title_style))
    elements.append(Paragraph(
        f"Generated on {datetime.date.today().strftime('%d %B %Y')} | Confidential",
        subtitle_style
    ))
    elements.append(Spacer(1, 10))

    # Applicant Info
    elements.append(Paragraph("Applicant Information", header_style))
    info_data = [
        ["Full Name", person_name],
        ["Person ID", person_id],
        ["Report Date", datetime.date.today().strftime('%d/%m/%Y')],
        ["Report Type", "Comprehensive Loan History"],
    ]
    info_table = Table(info_data, colWidths=[4 * cm, 12 * cm])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f9fafb')),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 15))

    # Loan Records
    elements.append(Paragraph("Loan Account Details", header_style))

    # Sample loan records
    sample_loans = [
        ["#", "Loan Type", "Amount ($1000s)", "Date", "Term", "Lender", "Status"],
        ["1", "Home Loan", "Rs. 25,00,000", "2019-06-15", "240 months", "SBI", "Repaid"],
        ["2", "Personal Loan", "Rs. 3,50,000", "2020-11-20", "36 months", "HDFC Bank", "Repaid"],
        ["3", "Car Loan", "Rs. 8,00,000", "2021-03-10", "60 months", "ICICI Bank", "Repaid"],
        ["4", "Education Loan", "Rs. 5,00,000", "2022-07-01", "48 months", "Axis Bank", "Active"],
        ["5", "Personal Loan", "Rs. 2,00,000", "2023-01-15", "24 months", "Kotak Mahindra", "Closed"],
    ]

    loan_table = Table(sample_loans, colWidths=[
        1 * cm, 3 * cm, 3.5 * cm, 2.5 * cm, 2.5 * cm, 3 * cm, 2 * cm
    ])
    loan_table.setStyle(TableStyle([
        # Header
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#A91D22')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        # Body
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ALIGN', (0, 1), (0, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
        # Status colors
        ('TEXTCOLOR', (6, 1), (6, 3), colors.HexColor('#16a34a')),  # Repaid = green
        ('TEXTCOLOR', (6, 5), (6, 5), colors.HexColor('#16a34a')),  # Closed = green
        ('TEXTCOLOR', (6, 4), (6, 4), colors.HexColor('#ca8a04')),  # Active = amber
        ('FONTNAME', (6, 1), (6, -1), 'Helvetica-Bold'),
    ]))
    elements.append(loan_table)
    elements.append(Spacer(1, 15))

    # Summary
    elements.append(Paragraph("Summary", header_style))
    summary_data = [
        ["Total Loans", "5"],
        ["Loans Repaid/Closed", "4"],
        ["Loans Defaulted", "0"],
        ["Loans Active", "1"],
        ["Repayment Rate", "100% (of completed loans)"],
        ["Average Loan Amount", "Rs. 8,70,000 ($87K)"],
        ["Overall Assessment", "Excellent borrowing track record"],
    ]
    summary_table = Table(summary_data, colWidths=[5 * cm, 11 * cm])
    summary_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f9fafb')),
        ('TEXTCOLOR', (1, -1), (1, -1), colors.HexColor('#16a34a')),
        ('FONTNAME', (1, -1), (1, -1), 'Helvetica-Bold'),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    # Disclaimer
    elements.append(Paragraph(
        "<i>This is a sample loan history report generated for demonstration purposes. "
        "In production, this document would be issued by a credit bureau (CIBIL, Experian, etc.).</i>",
        ParagraphStyle('Disclaimer', parent=normal_style,
                       fontSize=8, textColor=colors.grey)
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def generate_sample_pdf_simple() -> bytes:
    """
    Generate a simple text-based PDF without reportlab.
    Falls back to creating a readable text file as PDF.
    """
    # Use PyPDF2 to create a simple PDF with text
    # Since PyPDF2 can't easily write arbitrary text, we'll create
    # a minimal valid PDF manually
    content = """LOAN HISTORY REPORT
====================
Generated: {date}

APPLICANT: Rahul Sharma (PER00001)

LOAN ACCOUNT DETAILS:
---------------------
1. Home Loan - Amount: Rs. 25,00,000 - Date: 2019-06-15 - Status: Repaid - Lender: SBI
2. Personal Loan - Amount: Rs. 3,50,000 - Date: 2020-11-20 - Status: Repaid - Lender: HDFC Bank
3. Car Loan - Amount: Rs. 8,00,000 - Date: 2021-03-10 - Status: Repaid - Lender: ICICI Bank
4. Education Loan - Amount: Rs. 5,00,000 - Date: 2022-07-01 - Status: Active - Lender: Axis Bank
5. Personal Loan - Amount: Rs. 2,00,000 - Date: 2023-01-15 - Status: Closed - Lender: Kotak Mahindra

SUMMARY:
--------
Total Loans: 5
Repaid/Closed: 4
Defaulted: 0
Active: 1
Repayment Rate: 100%
Average Loan Amount: Rs. 8,70,000

Overall Assessment: Excellent borrowing track record.
""".format(date=datetime.date.today().strftime('%d/%m/%Y'))

    # Create a minimal PDF with the text content
    # Using raw PDF syntax for compatibility
    pdf_bytes = _create_minimal_pdf(content)
    return pdf_bytes


def _create_minimal_pdf(text: str) -> bytes:
    """Create a minimal valid PDF with the given text content."""
    lines = text.split('\n')
    # Build PDF content streams
    content_lines = []
    y = 800
    for line in lines:
        if y < 50:
            break
        # Escape special PDF characters
        escaped = line.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
        content_lines.append(f"BT /F1 10 Tf 50 {y} Td ({escaped}) Tj ET")
        y -= 14

    stream_content = "\n".join(content_lines)
    stream_length = len(stream_content)

    pdf = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj

2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj

3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj

4 0 obj
<< /Length {stream_length} >>
stream
{stream_content}
endstream
endobj

5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>
endobj

xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
0000000{str(317 + stream_length).zfill(4)} 00000 n 

trailer
<< /Size 6 /Root 1 0 R >>
startxref
0
%%EOF"""

    return pdf.encode('latin-1')


def generate_sample_pdf() -> bytes:
    """Generate a sample PDF — uses reportlab if available, else fallback."""
    if HAS_REPORTLAB:
        return generate_sample_pdf_reportlab()
    else:
        return generate_sample_pdf_simple()


if __name__ == "__main__":
    pdf_bytes = generate_sample_pdf()
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "sample_loan_report.pdf"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"✅ Sample PDF saved to: {output_path} ({len(pdf_bytes)} bytes)")

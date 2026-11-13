"""
FinSight AI - PDF & Statement Text Parser Utility
Extracts structured financial transactions from bank statement PDFs and raw text dumps.
"""
import io
import re
from typing import List, Dict, Any, Optional
import pandas as pd

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    try:
        import PyPDF2 as pypdf
        PYPDF_AVAILABLE = True
    except ImportError:
        PYPDF_AVAILABLE = False


def extract_text_from_pdf(file_bytes_or_path) -> str:
    """
    Extracts text content from a PDF file stream or file path using pypdf.
    """
    if not PYPDF_AVAILABLE:
        raise ImportError("pypdf or PyPDF2 is required to parse PDF statements. Please run `pip install pypdf`.")
        
    full_text = []
    
    if isinstance(file_bytes_or_path, (bytes, bytearray)):
        reader = pypdf.PdfReader(io.BytesIO(file_bytes_or_path))
    elif hasattr(file_bytes_or_path, 'read'):
        # Uploaded file object from Streamlit
        content = file_bytes_or_path.read()
        reader = pypdf.PdfReader(io.BytesIO(content))
    else:
        # File path
        reader = pypdf.PdfReader(str(file_bytes_or_path))
        
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            full_text.append(txt)
            
    return "\n".join(full_text)


def parse_statement_text(raw_text: str) -> pd.DataFrame:
    """
    Parses raw textual content from statements into standard tabular transactions.
    Looks for date patterns, description, transaction type, amounts, and balances.
    """
    lines = raw_text.splitlines()
    records: List[Dict[str, Any]] = []
    
    # Regex patterns for dates: YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY, etc.
    date_regex = re.compile(r'(\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b)')
    # Regex for currency/amounts: $1,234.56 or 1234.56
    amount_regex = re.compile(r'[-+]?\$?\d{1,3}(?:,\d{3})*\.\d{2}')
    
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
            
        date_match = date_regex.search(line_clean)
        if not date_match:
            continue
            
        date_str = date_match.group(1)
        # Find all amounts in this line
        amounts = amount_regex.findall(line_clean)
        if not amounts:
            continue
            
        # Extract values
        cleaned_amounts = []
        for amt in amounts:
            cleaned_num = float(amt.replace('$', '').replace(',', ''))
            cleaned_amounts.append(cleaned_num)
            
        # Determine amount and optional balance
        tx_amount = abs(cleaned_amounts[0])
        balance = abs(cleaned_amounts[1]) if len(cleaned_amounts) > 1 else 0.0
        
        # Determine debit vs credit
        line_lower = line_clean.lower()
        if 'credit' in line_lower or 'deposit' in line_lower or 'payroll' in line_lower or '+' in line_clean:
            tx_type = 'credit'
        else:
            tx_type = 'debit'
            
        # Extract description by stripping out the date and amounts
        desc = line_clean
        desc = desc.replace(date_str, '')
        for amt in amounts:
            desc = desc.replace(amt, '')
            
        # Clean up remaining keywords like "Debit", "Credit", etc.
        desc = re.sub(r'\b(debit|credit)\b', '', desc, flags=re.IGNORECASE)
        desc = re.sub(r'[\$|,|\|]', ' ', desc)
        desc = re.sub(r'\s+', ' ', desc).strip()
        
        if not desc:
            desc = "Transaction"
            
        records.append({
            "date": date_str,
            "description": desc,
            "amount": tx_amount,
            "type": tx_type,
            "balance": balance
        })
        
    df = pd.DataFrame(records)
    if not df.empty:
        # Normalize date to YYYY-MM-DD
        try:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        except Exception:
            pass
            
    return df


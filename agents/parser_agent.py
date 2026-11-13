"""
FinSight AI - Parser Agent
Responsible for ingesting, standardizing, and sanitizing statement files (CSV & PDF).
Transforms diverse bank statement formats into a unified transaction schema.
"""
import io
import re
from typing import Dict, Any, Tuple, Optional
import pandas as pd
from utils.pdf_parser import extract_text_from_pdf, parse_statement_text

class ParserAgent:
    """
    Parser Agent standardizes statements into uniform columns:
    ['date', 'description', 'amount', 'type', 'balance']
    """
    def __init__(self):
        self.column_synonyms = {
            'date': ['date', 'trans_date', 'transaction_date', 'tx_date', 'posted_date', 'booking_date'],
            'description': ['description', 'desc', 'merchant', 'payee', 'narrative', 'memo', 'details', 'name'],
            'amount': ['amount', 'amt', 'value', 'transaction_amount', 'total', 'charge'],
            'type': ['type', 'tx_type', 'transaction_type', 'credit_debit', 'dr_cr', 'd/c'],
            'balance': ['balance', 'running_balance', 'ending_balance', 'current_balance']
        }

    def parse_csv(self, file_source) -> pd.DataFrame:
        """
        Parses a CSV statement from a file path, string, or bytes stream.
        """
        if isinstance(file_source, (bytes, bytearray)):
            df = pd.read_csv(io.BytesIO(file_source))
        elif hasattr(file_source, 'read'):
            # Streamlit UploadedFile
            df = pd.read_csv(file_source)
        elif isinstance(file_source, str) and '\n' in file_source:
            df = pd.read_csv(io.StringIO(file_source))
        else:
            # File path
            df = pd.read_csv(str(file_source))
            
        return self._normalize_dataframe(df)

    def parse_pdf(self, file_source) -> pd.DataFrame:
        """
        Extracts and parses transactions from a PDF statement.
        """
        raw_text = extract_text_from_pdf(file_source)
        df = parse_statement_text(raw_text)
        return self._normalize_dataframe(df)

    def parse_statement(self, file_source, file_name: str = "") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Unified statement parser handling both CSV and PDF inputs.
        Returns (normalized_df, parse_metadata).
        """
        is_pdf = file_name.lower().endswith('.pdf') or (isinstance(file_source, str) and file_source.lower().endswith('.pdf'))
        
        try:
            if is_pdf:
                df = self.parse_pdf(file_source)
                parser_type = "PDF Extraction Engine"
            else:
                df = self.parse_csv(file_source)
                parser_type = "CSV Tabular Normalizer"
                
            metadata = {
                "success": True,
                "parser_used": parser_type,
                "row_count": len(df),
                "columns": list(df.columns),
                "error": None
            }
            return df, metadata
        except Exception as e:
            empty_df = pd.DataFrame(columns=['date', 'description', 'amount', 'type', 'balance'])
            metadata = {
                "success": False,
                "parser_used": "Error Fallback",
                "row_count": 0,
                "columns": [],
                "error": str(e)
            }
            return empty_df, metadata

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps arbitrary column names to standard fields and cleans types.
        """
        if df.empty:
            return pd.DataFrame(columns=['date', 'description', 'amount', 'type', 'balance'])
            
        df_norm = df.copy()
        
        # 1. Lowercase and strip existing column names
        df_norm.columns = [str(c).lower().strip().replace(' ', '_') for c in df_norm.columns]
        
        # 2. Map synonyms to standard keys
        col_mapping = {}
        for std_col, synonyms in self.column_synonyms.items():
            for c in df_norm.columns:
                if c in synonyms or any(syn in c for syn in synonyms):
                    col_mapping[c] = std_col
                    break
        df_norm = df_norm.rename(columns=col_mapping)
        
        # 3. Ensure required columns exist
        if 'description' not in df_norm.columns:
            for c in df_norm.columns:
                if df_norm[c].dtype == object and c != 'date':
                    df_norm['description'] = df_norm[c]
                    break
            if 'description' not in df_norm.columns:
                df_norm['description'] = 'Unspecified Transaction'
                
        if 'amount' not in df_norm.columns:
            # Check for Credit / Debit split columns
            credit_cols = [c for c in df_norm.columns if 'credit' in c or 'deposit' in c]
            debit_cols = [c for c in df_norm.columns if 'debit' in c or 'withdrawal' in c]
            
            if credit_cols and debit_cols:
                cr_col = credit_cols[0]
                dr_col = debit_cols[0]
                
                df_norm[cr_col] = pd.to_numeric(df_norm[cr_col].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce').fillna(0.0)
                df_norm[dr_col] = pd.to_numeric(df_norm[dr_col].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce').fillna(0.0)
                
                df_norm['amount'] = df_norm.apply(lambda r: r[cr_col] if r[cr_col] > 0 else r[dr_col], axis=1)
                df_norm['type'] = df_norm.apply(lambda r: 'credit' if r[cr_col] > 0 else 'debit', axis=1)
            else:
                df_norm['amount'] = 0.0
                
        if 'type' not in df_norm.columns:
            # Infer from amount sign if negative
            raw_amounts = pd.to_numeric(df_norm['amount'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce').fillna(0.0)
            df_norm['type'] = raw_amounts.apply(lambda x: 'debit' if x < 0 else 'credit')
            df_norm['amount'] = raw_amounts.abs()
        else:
            # Clean type column
            df_norm['type'] = df_norm['type'].astype(str).str.lower().str.strip()
            df_norm['type'] = df_norm['type'].apply(lambda x: 'credit' if any(k in x for k in ['cr', 'credit', 'dep', 'income']) else 'debit')
            
        if 'balance' not in df_norm.columns:
            df_norm['balance'] = 0.0
            
        # Clean numeric amount & balance
        df_norm['amount'] = pd.to_numeric(
            df_norm['amount'].astype(str).str.replace('$', '').str.replace(',', ''),
            errors='coerce'
        ).fillna(0.0).abs()
        
        df_norm['balance'] = pd.to_numeric(
            df_norm['balance'].astype(str).str.replace('$', '').str.replace(',', ''),
            errors='coerce'
        ).fillna(0.0)
        
        # Standardize date to YYYY-MM-DD
        if 'date' in df_norm.columns:
            df_norm['date'] = pd.to_datetime(df_norm['date'], errors='coerce').dt.strftime('%Y-%m-%d')
            df_norm['date'] = df_norm['date'].fillna('2024-01-01')
        else:
            df_norm['date'] = '2024-01-01'
            
        # Retain standard schema
        standard_cols = ['date', 'description', 'amount', 'type', 'balance']
        clean_df = df_norm[standard_cols].copy()
        clean_df['description'] = clean_df['description'].astype(str).str.strip()
        
        return clean_df.reset_index(drop=True)


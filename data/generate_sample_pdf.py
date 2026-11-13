"""
Utility to generate a sample bank statement PDF for FinSight AI testing and evaluation.
Uses standard FPDF to generate a realistic First National Bank statement.
"""
import os
import sys

try:
    from fpdf import FPDF
except ImportError:
    print("fpdf package not installed. Skipping PDF generation.")
    sys.exit(0)

class StatementPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'FIRST HORIZON NATIONAL BANK', border=0, ln=1, align='L')
        self.set_font('Helvetica', '', 9)
        self.cell(0, 5, 'Account Statement | Checking Account: ****4921', border=0, ln=1, align='L')
        self.cell(0, 5, 'Statement Period: Jan 01, 2024 - Jan 31, 2024', border=0, ln=1, align='L')
        self.ln(5)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_sample_pdf(output_path: str = "sample_statement.pdf"):
    pdf = StatementPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Account Summary Block
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'Account Summary', ln=1)
    pdf.set_font('Helvetica', '', 9)
    pdf.cell(50, 6, 'Starting Balance: $6,500.00', ln=0)
    pdf.cell(50, 6, 'Total Deposits: $7,000.00', ln=0)
    pdf.cell(50, 6, 'Total Withdrawals: $2,936.62', ln=1)
    pdf.ln(4)
    
    # Table Header
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(230, 235, 245)
    pdf.cell(25, 7, 'Date', 1, 0, 'C', 1)
    pdf.cell(85, 7, 'Description', 1, 0, 'L', 1)
    pdf.cell(25, 7, 'Type', 1, 0, 'C', 1)
    pdf.cell(25, 7, 'Amount', 1, 0, 'R', 1)
    pdf.cell(30, 7, 'Balance', 1, 1, 'R', 1)
    
    pdf.set_font('Helvetica', '', 8)
    
    sample_rows = [
        ("2024-01-01", "Acme Corp Payroll Direct Deposit", "Credit", "3500.00", "6500.00"),
        ("2024-01-02", "Apex Property Management Rent", "Debit", "1450.00", "5050.00"),
        ("2024-01-03", "Con Edison Electric & Gas", "Debit", "94.20", "4955.80"),
        ("2024-01-04", "Whole Foods Market Groceries", "Debit", "134.80", "4821.00"),
        ("2024-01-05", "Shell Oil Gas Station", "Debit", "48.50", "4772.50"),
        ("2024-01-07", "Netflix Monthly Subscription", "Debit", "15.49", "4757.01"),
        ("2024-01-08", "Starbucks Coffee #1042", "Debit", "6.75", "4750.26"),
        ("2024-01-09", "CVS Pharmacy Prescription", "Debit", "32.40", "4717.86"),
        ("2024-01-10", "Amazon.com Retail Purchase", "Debit", "64.20", "4653.66"),
        ("2024-01-12", "Trader Joe's Supermarket", "Debit", "82.40", "4571.26"),
        ("2024-01-14", "Uber Trip San Francisco", "Debit", "24.50", "4546.76"),
        ("2024-01-15", "Acme Corp Payroll Direct Deposit", "Credit", "3500.00", "8046.76"),
        ("2024-01-16", "Vanguard Index Fund ETF", "Debit", "500.00", "7546.76"),
        ("2024-01-17", "Verizon Fios Internet Service", "Debit", "69.99", "7476.77"),
        ("2024-01-18", "Chipotle Mexican Grill", "Debit", "18.25", "7458.52"),
        ("2024-01-20", "Target Retail Store", "Debit", "92.15", "7366.37"),
        ("2024-01-22", "Spotify Premium Music", "Debit", "11.99", "7354.38"),
        ("2024-01-24", "Chevron Service Station", "Debit", "45.00", "7309.38"),
        ("2024-01-26", "Bella Italia Trattoria", "Debit", "88.40", "7220.98"),
        ("2024-01-28", "Equinox Fitness Club Dues", "Debit", "45.00", "7175.98"),
        ("2024-01-30", "Safeway Groceries", "Debit", "112.60", "7063.38"),
    ]
    
    for row in sample_rows:
        pdf.cell(25, 6, row[0], 1, 0, 'C')
        pdf.cell(85, 6, row[1], 1, 0, 'L')
        pdf.cell(25, 6, row[2], 1, 0, 'C')
        pdf.cell(25, 6, f"${float(row[3]):,.2f}", 1, 0, 'R')
        pdf.cell(30, 6, f"${float(row[4]):,.2f}", 1, 1, 'R')
        
    pdf.output(output_path, 'F')
    print(f"Generated PDF statement at: {output_path}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(script_dir, "sample_statement.pdf")
    create_sample_pdf(target_path)


#!/usr/bin/env python3
"""
Konvertiert Markdown-Dateien zu PDF
"""

import markdown
from weasyprint import HTML, CSS
from pathlib import Path
import sys

def markdown_to_pdf(input_file, output_file=None):
    """
    Konvertiert eine Markdown-Datei zu PDF
    
    Parameters:
    - input_file: Pfad zur Markdown-Datei
    - output_file: Pfad zur Ausgabe-PDF (optional, Standard: input_file.pdf)
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Fehler: Datei '{input_file}' nicht gefunden!")
        return False
    
    # Output-Dateiname generieren
    if output_file is None:
        output_file = input_path.with_suffix('.pdf')
    else:
        output_file = Path(output_file)
    
    print(f"Konvertiere '{input_file}' zu PDF...")
    
    try:
        # Markdown-Datei lesen
        with open(input_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Markdown zu HTML konvertieren
        html_content = markdown.markdown(
            markdown_content,
            extensions=['extra', 'codehilite', 'tables', 'toc']
        )
        
        # HTML mit CSS-Styling wrappen
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                @page {{
                    size: A4;
                    margin: 2cm;
                }}
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    font-size: 11pt;
                    line-height: 1.6;
                    color: #333;
                }}
                h1 {{
                    font-size: 24pt;
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                    page-break-after: avoid;
                }}
                h2 {{
                    font-size: 18pt;
                    color: #34495e;
                    border-bottom: 2px solid #95a5a6;
                    padding-bottom: 8px;
                    margin-top: 25px;
                    page-break-after: avoid;
                }}
                h3 {{
                    font-size: 14pt;
                    color: #555;
                    margin-top: 20px;
                    page-break-after: avoid;
                }}
                h4 {{
                    font-size: 12pt;
                    color: #666;
                    margin-top: 15px;
                }}
                code {{
                    background-color: #f4f4f4;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                    font-size: 10pt;
                }}
                pre {{
                    background-color: #f4f4f4;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    overflow-x: auto;
                    page-break-inside: avoid;
                }}
                pre code {{
                    background-color: transparent;
                    padding: 0;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 15px 0;
                    page-break-inside: avoid;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                ul, ol {{
                    margin: 10px 0;
                    padding-left: 30px;
                }}
                li {{
                    margin: 5px 0;
                }}
                blockquote {{
                    border-left: 4px solid #3498db;
                    margin: 15px 0;
                    padding-left: 15px;
                    color: #555;
                    font-style: italic;
                }}
                a {{
                    color: #3498db;
                    text-decoration: none;
                }}
                hr {{
                    border: none;
                    border-top: 2px solid #ecf0f1;
                    margin: 20px 0;
                }}
                .toc {{
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    padding: 15px;
                    margin: 20px 0;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # HTML zu PDF konvertieren
        HTML(string=html_template).write_pdf(output_file)
        
        print(f"✓ Erfolgreich erstellt: '{output_file}'")
        return True
        
    except Exception as e:
        print(f"Fehler bei der Konvertierung: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Verwendung: python markdown_to_pdf.py <input.md> [output.pdf]")
        print("\nBeispiel:")
        print("  python markdown_to_pdf.py Data_Science_Versicherung_Interview.md")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = markdown_to_pdf(input_file, output_file)
    sys.exit(0 if success else 1)


#!/usr/bin/env python3
"""
Konvertiert Markdown-Dateien zu HTML (zum Drucken/Export als PDF)
"""

import markdown
from pathlib import Path
import sys

def markdown_to_html(input_file, output_file=None):
    """
    Konvertiert eine Markdown-Datei zu HTML (print-ready)
    
    Parameters:
    - input_file: Pfad zur Markdown-Datei
    - output_file: Pfad zur Ausgabe-HTML (optional)
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Fehler: Datei '{input_file}' nicht gefunden!")
        return False
    
    # Output-Dateiname generieren
    if output_file is None:
        output_file = input_path.with_suffix('.html')
    else:
        output_file = Path(output_file)
    
    print(f"Konvertiere '{input_file}' zu HTML...")
    
    try:
        # Markdown-Datei lesen
        with open(input_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Markdown zu HTML konvertieren
        html_content = markdown.markdown(
            markdown_content,
            extensions=['extra', 'codehilite', 'tables', 'toc']
        )
        
        # HTML mit CSS-Styling wrappen (print-optimiert)
        html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Science Versicherung - Vorstellungsgespräch Vorbereitung</title>
    <style>
        @media print {{
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{
                font-size: 11pt;
            }}
            h1, h2, h3, h4 {{
                page-break-after: avoid;
            }}
            pre, blockquote {{
                page-break-inside: avoid;
            }}
        }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 12pt;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
        }}
        h1 {{
            font-size: 28pt;
            color: #2c3e50;
            border-bottom: 4px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 30px;
            page-break-after: avoid;
        }}
        h2 {{
            font-size: 20pt;
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 10px;
            margin-top: 35px;
            margin-bottom: 20px;
            page-break-after: avoid;
        }}
        h3 {{
            font-size: 16pt;
            color: #555;
            margin-top: 25px;
            margin-bottom: 15px;
            page-break-after: avoid;
        }}
        h4 {{
            font-size: 14pt;
            color: #666;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', 'Consolas', monospace;
            font-size: 11pt;
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
            display: block;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            page-break-inside: avoid;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
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
            margin: 15px 0;
            padding-left: 35px;
        }}
        li {{
            margin: 8px 0;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 20px 0;
            padding-left: 20px;
            color: #555;
            font-style: italic;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }}
        .toc {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            padding: 20px;
            margin: 30px 0;
            border-radius: 5px;
        }}
        .print-instructions {{
            background-color: #fff3cd;
            border: 2px solid #ffc107;
            padding: 15px;
            margin-bottom: 30px;
            border-radius: 5px;
        }}
        @media print {{
            .print-instructions {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="print-instructions">
        <strong>📄 Als PDF speichern:</strong><br>
        Drücke <strong>Strg+P</strong> (oder <strong>Cmd+P</strong> auf Mac) und wähle "Als PDF speichern" als Drucker.<br>
        Oder: Rechtsklick → "Drucken" → "Als PDF speichern"
    </div>
    {html_content}
</body>
</html>"""
        
        # HTML-Datei schreiben
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        print(f"[OK] HTML erstellt: '{output_file}'")
        print(f"  Oeffne die Datei im Browser und drucke sie als PDF (Strg+P)")
        return True
        
    except Exception as e:
        print(f"Fehler bei der Konvertierung: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Verwendung: python markdown_to_html.py <input.md> [output.html]")
        print("\nBeispiel:")
        print("  python markdown_to_html.py Data_Science_Versicherung_Interview.md")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = markdown_to_html(input_file, output_file)
    sys.exit(0 if success else 1)


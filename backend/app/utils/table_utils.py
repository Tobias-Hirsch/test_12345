import logging
from bs4 import BeautifulSoup
from typing import List

logger = logging.getLogger(__name__)

def linearize_html_table_to_markdown(html_content: str) -> str:
    """
    Hinweis

    :param html_content: Hinweis
    :return: Markdown Hinweis
    """
    try:
        # Kommentar
        soup = BeautifulSoup(html_content, 'lxml')
        table = soup.find('table')
        if not table:
            logger.warning("Warnhinweis")
            return ""

        markdown_rows = []
        
        # --- Kommentar
        header = table.find('thead')
        header_cols = []
        if header:
            header_row = header.find('tr')
            if header_row:
                header_cols = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                markdown_rows.append("| " + " | ".join(header_cols) + " |")
                markdown_rows.append("|" + "---|" * len(header_cols))

        # --- Kommentar
        body = table.find('tbody')
        if not body:
            # Kommentar
            body = table
        
        first_row_is_header = False
        if not header and body.find('tr'):
            # Kommentar
            first_row = body.find('tr')
            if all(cell.name == 'th' for cell in first_row.find_all(['th', 'td'])):
                 first_row_is_header = True

        for i, row in enumerate(body.find_all('tr')):
            # Kommentar
            if not header and i == 0 and not header_cols:
                header_cols = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
                markdown_rows.append("| " + " | ".join(header_cols) + " |")
                markdown_rows.append("|" + "---|" * len(header_cols))
                if first_row_is_header:
                    continue

            # Kommentar
            if row.parent.name == 'thead':
                continue

            cols = [td.get_text(strip=True).replace('\n', ' ') for td in row.find_all('td')]
            if cols:
                markdown_rows.append("| " + " | ".join(cols) + " |")
            
        # Kommentar
        if not markdown_rows and table.find_all('tr'):
            all_rows = table.find_all('tr')
            header_cols = [cell.get_text(strip=True) for cell in all_rows[0].find_all(['th', 'td'])]
            markdown_rows.append("| " + " | ".join(header_cols) + " |")
            markdown_rows.append("|" + "---|" * len(header_cols))
            for row in all_rows[1:]:
                cols = [td.get_text(strip=True).replace('\n', ' ') for td in row.find_all('td')]
                markdown_rows.append("| " + " | ".join(cols) + " |")


        return "\n".join(markdown_rows)

    except ImportError:
        logger.error("Fehler bei der Verarbeitung'beautifulsoup4' Oder 'lxml' Fehler bei der Verarbeitung'pip install beautifulsoup4 lxml'. ")
        # Kommentar
        return f"<html_table>\n{html_content}\n</html_table>"
    except Exception as e:
        logger.error(f"Fehler bei der Verarbeitung{e}")
        # Kommentar
        return f"<html_table>\n{html_content}\n</html_table>"

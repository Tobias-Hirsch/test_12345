# Developer: Jinglu Han
# mailbox: admin@de-manufacturing.cn

import aiohttp
import pandas as pd
import io
import asyncio
from typing import Dict, List, Any, Union
from app.services.logging import xlsx_process_content_logger as logger


async def download_and_parse_xlsx(url) -> \
Union[Dict[str, pd.DataFrame], None]:
    """
    Hinweis

    Args:
        url: xlsxHinweis

    Returns:
        Hinweis
        Hinweis
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Fehler bei der Verarbeitung{response.status}")
                    return None

                # Kommentar
                logger.info("Hinweis")
                content = await response.read()

                # Kommentar
                logger.info("Hinweis")
                excel_data = {}
                with io.BytesIO(content) as file_obj:
                    # Kommentar
                    excel_file = pd.ExcelFile(file_obj, engine='openpyxl')
                    sheet_names = excel_file.sheet_names

                    for sheet_name in sheet_names:
                        df = excel_file.parse(sheet_name)
                        excel_data[sheet_name] = df
                result :str  = ''
                for sheet_name, df in excel_data.items():
                    result += f"Arbeitsblatt: {sheet_name}\n"
                    result += df.to_string()
                    result += "\n"
                    result += '-' * 50
                    result += "\n"
                return result
    except Exception as e:
        logger.error(f"Fehler bei der Excel-Verarbeitung: {str(e)}")
        return None

async def extract_excel_content_from_file(file_path: str) -> Union[str, None]:
    """
    Hinweis

    Args:
        file_path: Hinweis

    Returns:
        Hinweis
    """
    try:
        logger.info(f"Hinweis{file_path}")
        excel_data = {}
        # Use pandas to read the Excel file directly from the local path
        excel_file = pd.ExcelFile(file_path, engine='openpyxl')
        sheet_names = excel_file.sheet_names

        for sheet_name in sheet_names:
            df = excel_file.parse(sheet_name)
            excel_data[sheet_name] = df

        result: str = ''
        for sheet_name, df in excel_data.items():
            result += f"Arbeitsblatt: {sheet_name}\n"
            result += df.to_string()
            result += "\n"
            result += '-' * 50
            result += "\n"
        logger.info(f"Hinweis{file_path} Hinweis")
        return result
    except Exception as e:
        logger.error(f"Fehler bei der Verarbeitung{str(e)}")
        return None

async def extract_excel_content_from_bytes(file_content: bytes, filename: str = None) -> Union[str, None]:
    """
    Hinweis

    Args:
        file_content: ExcelHinweis
        filename: Dateiname, Hinweis

    Returns:
        Hinweis
    """
    try:
        logger.info("Hinweis")
        
        # Kommentar
        engine = 'openpyxl'  # Hinweis
        if filename:
            file_ext = filename.lower().split('.')[-1]
            if file_ext == 'xls':
                engine = 'xlrd'
                logger.info("Hinweis")
            else:
                logger.info("Hinweis")
        
        excel_data = {}
        with io.BytesIO(file_content) as file_obj:
            try:
                # Kommentar
                excel_file = pd.ExcelFile(file_obj, engine=engine)
                sheet_names = excel_file.sheet_names

                for sheet_name in sheet_names:
                    df = excel_file.parse(sheet_name)
                    excel_data[sheet_name] = df
                    
            except Exception as engine_error:
                logger.warning(f"Fehler bei der Verarbeitung{engine}Fehler bei der Verarbeitung{engine_error}, Fehler bei der Verarbeitung")
                # Kommentar
                file_obj.seek(0)  # Hinweis
                excel_file = pd.ExcelFile(file_obj)  # Hinweis
                sheet_names = excel_file.sheet_names

                for sheet_name in sheet_names:
                    df = excel_file.parse(sheet_name)
                    excel_data[sheet_name] = df

        result: str = ''
        for sheet_name, df in excel_data.items():
            result += f"Arbeitsblatt: {sheet_name}\n"
            result += df.to_string()
            result += "\n"
            result += '-' * 50
            result += "\n"
        
        logger.info(f"Hinweis{len(excel_data)}Hinweis")
        return result
    except Exception as e:
        logger.error(f"Fehler bei der Verarbeitung{str(e)}")
        return None

# Kommentar
async def main():
    result = await download_and_parse_xlsx('http://localhost:9001/api/v1/download-shared-object/aHR0cDovLzEyNy4wLjAuMTo5MDAwL21tLXJhZy1idWNrZXQvcm9zdGkvQVdJX0ZMNDFBU1MwMDAyLXh4LUJfQTEueGxzeD9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPTI5NFlTOUlEWVVGNjYxVExBV0NNJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDE0MjYyOFomWC1BbXotRXhwaXJlcz00MzIwMCZYLUFtei1TZWN1cml0eS1Ub2tlbj1leUpoYkdjaU9pSklVelV4TWlJc0luUjVjQ0k2SWtwWFZDSjkuZXlKaFkyTmxjM05MWlhraU9pSXlPVFJaVXpsSlJGbFZSalkyTVZSTVFWZERUU0lzSW1WNGNDSTZNVGMwT0RNeE1EZ3hOQ3dpY0dGeVpXNTBJam9pYldsdWFXOWhaRzFwYmlKOS5VeGtZY05jRUxtTmp5aUNPNWgwdFc4Sy1KTUVOaWwxTzQ1dUp3YXc4TmRxRWkzWTlISDFGdHVrUXROdFlheGpYdHhHUERWOGJkZHh5RDNaZ1VCMWQtQSZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmdmVyc2lvbklkPW51bGwmWC1BbXotU2lnbmF0dXJlPWUzYWIxMzgxN2Q4MWY1ZTk3ZjA3MjFjZTczYTY0ZGJhOThmYWRhNTZlMWYyMjRhNzViODY2NTYxMWFkZjc1NDg')
    if result:
        for sheet_name, df in result.items():
            print(f"Arbeitsblatt: {sheet_name}")
            print(df.head())
            print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())
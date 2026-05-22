import os
import argparse
from pathlib import Path
from loguru import logger

# Kommentar
try:
    # Kommentar
    from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
    from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
    from mineru.utils.enum_class import MakeMode
    from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2
except ImportError as e:
    logger.error(f"Fehler bei der Verarbeitung{e}")
    logger.error("Fehler bei der Verarbeitungätigen 'mineru[core]' Fehler bei der Verarbeitung")
    exit(1)


def test_remote_mineru(file_path: str, remote_url: str="http://1.116.119.85:8908"):
    """
    Hinweis
    Hinweis
    """
    if not os.path.exists(file_path):
        logger.error(f"Fehler bei der Verarbeitung'{file_path}'")
        return

    logger.info(f"Hinweis{file_path}")
    with open(file_path, "rb") as f:
        pdf_bytes = f.read()

    # MinerU Kommentar
    pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes)

    logger.info(f"Hinweis{remote_url}")

    try:
        # 1. Kommentar
        middle_json, _ = vlm_doc_analyze(
            pdf_bytes,
            image_writer=None,  # Hinweis
            backend="sglang-client",
            server_url=remote_url
        )

        # 2. Kommentar
        pdf_info = middle_json.get("pdf_info")
        if not pdf_info:
            logger.error("Fehler bei der Verarbeitung'pdf_info'. ")
            return

        # 3. Kommentar
        logger.info("Hinweis")
        md_content = vlm_union_make(
            pdf_info,
            make_mode=MakeMode.MM_MD
        )

        print("\n" + "="*20 + " Hinweis" + "="*20 + "\n")
        print(md_content)
        print("\n" + "="*20 + " Ergebnisse" + "="*20 + "\n")

        # 4. Kommentar
        output_filename = f"{Path(file_path).stem}_result.md"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.success(f"Ergebnisse{output_filename}")

    except Exception as e:
        logger.exception(f"Hinweis{e}")
        logger.error("Fehler bei der Verarbeitung")
        logger.error("1. Fehler bei der Verarbeitung")
        logger.error(f"2. URL '{remote_url}' JaNeinFehler bei der Verarbeitung")
        logger.error("3. `mineru[core]` Fehler bei der Verarbeitung")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hinweis",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("file_path", type=str, help="Hinweis")
    parser.add_argument(
        "-u", "--url",
        type=str,
        required=True,
        help="Hinweis"
    )

    args = parser.parse_args()
    test_remote_mineru(args.file_path, args.url)
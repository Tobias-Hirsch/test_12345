"""
Hinweis
Hinweis

Hinweis
1. Hinweis
2. Hinweis
3. Hinweis
4. Hinweis

Hinweis
Hinweis
"""

import logging
import asyncio
import aiohttp
import urllib.parse
import json
import base64
import time
from typing import Optional, Dict, Any, List, Tuple

from app.core.config import settings

logger = logging.getLogger(__name__)

# Kommentar
try:
    from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
    from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
    from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
    from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
    from mineru.utils.enum_class import MakeMode
    from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
    from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2
    MINERU_INSTALLED = True
    logger.info("MinerUHinweis")
except ImportError as e:
    logger.warning(f"MinerUWarnhinweis{e}")
    logger.warning("Warnhinweis")
    MINERU_INSTALLED = False
    
    # Kommentar
    def convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes: bytes) -> bytes:
        """Hinweis"""
        return pdf_bytes


class HybridVLMProcessor:
    """
    Hinweis
    Hinweis
    1. Hinweis
    """
    
    def __init__(self):
        """Hinweis"""
        self.server_url = settings.MINERU_SGLANG_SERVER_URL
        self.modes = {
            "remote_sglang": False,
            "local_vlm": MINERU_INSTALLED,
            "local_pipeline": MINERU_INSTALLED
        }
        
        # Kommentar
        if self.server_url:
            parsed_url = urllib.parse.urlparse(self.server_url)
            self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            logger.info(f"Hinweis{self.base_url}")
        else:
            self.base_url = None
            logger.info("Nicht konfiguriertHinweis")
        
        # Kommentar
        self.timeout_seconds = 300  # 5Hinweis
        self.max_retries = 2
        self.retry_delay = 1.0
        
        logger.info(f"HybridVLMProcessor Hinweis")
        logger.info(f"Hinweis{[k for k, v in self.modes.items() if v]}")
    
    async def _test_remote_server(self) -> bool:
        """Hinweis"""
        if not self.base_url:
            return False
        
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.base_url}/get_model_info") as response:
                    if response.status == 200:
                        logger.info("Hinweis")
                        self.modes["remote_sglang"] = True
                        return True
        except Exception as e:
            logger.warning(f"Warnhinweis{e}")
        
        self.modes["remote_sglang"] = False
        return False
    
    async def _process_with_remote_sglang(self, file_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """Hinweis"""
        if not self.modes["remote_sglang"]:
            return None
        
        logger.info(f"Hinweis{filename}")
        
        # Kommentar
        # OderKommentar
        try:
            if MINERU_INSTALLED:
                # Kommentar
                processed_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(file_bytes)
                
                middle_json, _ = vlm_doc_analyze(
                    processed_bytes,
                    image_writer=None,
                    backend="sglang-client",
                    server_url=self.base_url
                )
                
                pdf_info = middle_json.get("pdf_info")
                if pdf_info:
                    content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, "")
                    if content_list:
                        logger.info(f"Hinweis{filename} - {len(content_list)} Einträge")
                        return {"result": content_list}
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung{filename} - {e}")
        
        return None
    
    def _process_with_local_vlm(self, file_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """Hinweis"""
        if not self.modes["local_vlm"]:
            return None
        
        logger.info(f"Hinweis{filename}")
        
        try:
            # Kommentar
            processed_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(file_bytes)
            
            # Kommentar
            middle_json, _ = vlm_doc_analyze(
                processed_bytes,
                image_writer=None,
                backend="local",  # OderHinweis
                server_url=None
            )
            
            pdf_info = middle_json.get("pdf_info")
            if pdf_info:
                content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, "")
                if content_list:
                    logger.info(f"Hinweis{filename} - {len(content_list)} Einträge")
                    return {"result": content_list}
                    
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung{filename} - {e}")
        
        return None
    
    def _process_with_local_pipeline(self, file_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """Hinweis"""
        if not self.modes["local_pipeline"]:
            return None
        
        logger.info(f"Hinweis{filename}")
        
        try:
            # Kommentar
            processed_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(file_bytes)
            
            # Kommentar
            infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                [processed_bytes],
                ['ch'],  # Hinweis
                parse_method="auto",
                formula_enable=True,
                table_enable=True
            )
            
            if infer_results and infer_results[0]:
                # Kommentar
                middle_json = pipeline_result_to_middle_json(
                    model_list=infer_results[0],
                    images_list=all_image_lists[0],
                    pdf_doc=all_pdf_docs[0],
                    image_writer=None,
                    lang=lang_list[0],
                    ocr_enable=ocr_enabled_list[0]
                )
                
                pdf_info = middle_json.get("pdf_info")
                if pdf_info:
                    content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, "")
                    if content_list:
                        logger.info(f"Hinweis{filename} - {len(content_list)} Einträge")
                        return {"result": content_list}
                        
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung{filename} - {e}")
        
        return None
    
    async def process_document_bytes(self, file_bytes: bytes, filename: str, strategy: str = "auto") -> Optional[Dict[str, Any]]:
        """
        Hinweis
        
        Args:
            file_bytes: Dateibytes
            filename: Dateiname
            strategy: Hinweis"auto", "remote", "local-vlm", "pipeline")
            
        Returns:
            Hinweis{"result": [...]} Oder None
        """
        if not file_bytes:
            logger.error(f"Dateidaten sind leer: {filename}")
            return None
        
        file_size_mb = len(file_bytes) / 1024 / 1024
        logger.info(f"Hinweis{filename} ({file_size_mb:.2f} MB) - Hinweis{strategy}")
        
        # Kommentar
        if strategy == "remote" and self.base_url:
            if await self._test_remote_server():
                return await self._process_with_remote_sglang(file_bytes, filename)
            else:
                logger.warning("Warnhinweis")
        
        elif strategy == "local-vlm":
            return await asyncio.to_thread(self._process_with_local_vlm, file_bytes, filename)
        
        elif strategy == "pipeline":
            return await asyncio.to_thread(self._process_with_local_pipeline, file_bytes, filename)
        
        # KommentarätKommentar
        processing_methods = [
            ("Hinweis", self._try_remote_processing),
            ("Hinweis", self._try_local_vlm_processing),
            ("Hinweis", self._try_local_pipeline_processing)
        ]
        
        for method_name, method_func in processing_methods:
            try:
                result = await method_func(file_bytes, filename)
                if result:
                    logger.info(f"Hinweis{method_name} - {filename}")
                    return result
                else:
                    logger.debug(f"{method_name} Hinweis")
            except Exception as e:
                logger.warning(f"{method_name} Ausnahme bei Verarbeitung: {e}")
        
        logger.error(f"Fehler bei der Verarbeitung{filename}")
        return None
    
    async def _try_remote_processing(self, file_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """Hinweis"""
        if self.base_url and await self._test_remote_server():
            return await self._process_with_remote_sglang(file_bytes, filename)
        return None
    
    async def _try_local_vlm_processing(self, file_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """Hinweis"""
        if self.modes["local_vlm"]:
            return await asyncio.to_thread(self._process_with_local_vlm, file_bytes, filename)
        return None
    
    async def _try_local_pipeline_processing(self, file_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """Hinweis"""
        if self.modes["local_pipeline"]:
            return await asyncio.to_thread(self._process_with_local_pipeline, file_bytes, filename)
        return None


class FallbackProcessor:
    """Hinweis"""
    
    def __init__(self):
        logger.warning("FallbackProcessor Warnhinweis")
    
    async def process_document_bytes(self, file_bytes: bytes, filename: str, strategy: str = "fallback") -> Optional[Dict[str, Any]]:
        """Hinweis"""
        logger.error(f"Fehler bei der Verarbeitung{filename} - MinerUFehler bei der Verarbeitung")
        logger.error("Fehler bei der Verarbeitung")
        return {"result": [{"type": "text", "text": f"DokumenteHinweis{filename}"}]}


# Kommentar
_hybrid_processor = None

def get_hybrid_mineru_processor():
    """
    Hinweis
    Hinweis
    """
    global _hybrid_processor
    
    if _hybrid_processor is None:
        if MINERU_INSTALLED or settings.MINERU_SGLANG_SERVER_URL:
            logger.info("Hinweis")
            _hybrid_processor = HybridVLMProcessor()
        else:
            logger.warning("Warnhinweis")
            _hybrid_processor = FallbackProcessor()
    
    return _hybrid_processor


# Kommentar
mineru_client = get_hybrid_mineru_processor()

# Kommentar
__all__ = [
    "get_hybrid_mineru_processor",
    "mineru_client",
    "HybridVLMProcessor",
    "FallbackProcessor"
]
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

import asyncio
import aiohttp
import json
import logging
import time
import base64
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import urllib.parse

from app.core.config import settings

logger = logging.getLogger(__name__)

class OptimizedMinerUVLMClient:
    """
    Hinweis
    Hinweis
    """
    
    def __init__(self, server_url: Optional[str] = None):
        """
        Hinweis
        
        Args:
            server_url: SGLangHinweis
        """
        self.server_url = server_url or settings.MINERU_SGLANG_SERVER_URL
        if not self.server_url:
            raise ValueError("MINERU_SGLANG_SERVER_URL Nicht konfiguriert")
        
        # Kommentar
        parsed_url = urllib.parse.urlparse(self.server_url)
        self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Kommentaräge
        self.timeout = aiohttp.ClientTimeout(total=600)  # 10Hinweis
        self.max_retries = 3
        self.retry_delay = 2.0  # Hinweis
        
        logger.info(f"Hinweis{self.base_url}")
    
    async def _create_session(self) -> aiohttp.ClientSession:
        """Hinweis"""
        return aiohttp.ClientSession(
            timeout=self.timeout,
            connector=aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                keepalive_timeout=60,
                enable_cleanup_closed=True
            ),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "RostiAI-MinerU-VLM-Client/1.0"
            }
        )
    
    async def check_server_health(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Hinweis
        
        Returns:
            (is_healthy, server_info)
        """
        health_endpoints = ["/health", "/api/health", "/api/v1/health", "/status"]
        
        async with await self._create_session() as session:
            for endpoint in health_endpoints:
                try:
                    async with session.get(f"{self.base_url}{endpoint}") as response:
                        if response.status == 200:
                            try:
                                data = await response.json()
                                logger.info(f"Server-Healthcheck erfolgreich: {endpoint}")
                                return True, data
                            except:
                                logger.info(f"Hinweis{endpoint}")
                                return True, {"status": "ok", "endpoint": endpoint}
                except Exception as e:
                    logger.debug(f"Hinweis{endpoint}: {e}")
                    continue
        
        logger.warning("Warnhinweis")
        return False, {}
    
    async def _discover_parse_endpoint(self, session: aiohttp.ClientSession) -> Optional[str]:
        """
        Hinweis
        
        Returns:
            Hinweis
        """
        candidate_endpoints = [
            "/api/v1/parse_pdf",
            "/api/v1/vlm/parse", 
            "/api/v1/parse",
            "/api/parse",
            "/parse",
            "/api/v1/document/parse",
            "/api/v1/mineru/parse"
        ]
        
        for endpoint in candidate_endpoints:
            try:
                # SendenKommentar
                test_data = {"test": "endpoint_discovery"}
                async with session.post(f"{self.base_url}{endpoint}", json=test_data) as response:
                    # Kommentar
                    if response.status != 404:
                        logger.info(f"Hinweis{endpoint} (Status: {response.status})")
                        return endpoint
                        
            except Exception as e:
                logger.debug(f"Hinweis{endpoint}: {e}")
                continue
        
        logger.warning("Warnhinweis")
        return None
    
    def _prepare_request_data(self, pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Hinweis
        Hinweis
        
        Args:
            pdf_bytes: PDFDateibytes
            filename: Dateiname
            
        Returns:
            Hinweis
        """
        pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')
        
        # Kommentar
        return {
            "file_data": pdf_b64,
            "filename": filename,
            "mode": "vlm",
            "backend": "sglang",
            "config": {
                "formula_enable": True,
                "table_enable": True,
                "parse_method": "auto",
                "lang": "auto",  # Hinweis
                "output_format": "json"
            },
            "options": {
                "extract_images": False,  # Hinweis
                "extract_tables": True,
                "extract_formulas": True,
                "preserve_layout": True
            }
        }
    
    def _prepare_alternative_formats(self, pdf_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Hinweis
        
        Returns:
            Hinweis
        """
        pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')
        
        return [
            # Kommentar
            {
                "pdf_data": pdf_b64,
                "filename": filename,
                "vlm_mode": True
            },
            
            # Kommentar
            {
                "document": pdf_b64,
                "name": filename,
                "type": "pdf",
                "backend": "vlm-sglang",
                "parse_options": {
                    "formula": True,
                    "table": True
                }
            },
            
            # Kommentar
            {
                "file": pdf_b64,
                "filename": filename,
                "parser": "mineru-vlm",
                "settings": {
                    "mode": "vlm",
                    "engine": "sglang"
                }
            }
        ]
    
    async def _parse_with_endpoint(
        self, 
        session: aiohttp.ClientSession, 
        endpoint: str, 
        request_data: Dict[str, Any],
        filename: str
    ) -> Optional[Dict[str, Any]]:
        """
        Hinweis
        
        Args:
            session: HTTPHinweis
            endpoint: APIHinweis
            request_data: Hinweis
            filename: Dateiname(Hinweis
            
        Returns:
            Hinweis
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            logger.info(f"Hinweis{filename} -> {endpoint}")
            start_time = time.time()
            
            async with session.post(url, json=request_data) as response:
                processing_time = time.time() - start_time
                
                logger.info(f"APIHinweis{response.status} (Dauer: {processing_time:.2f}s)")
                
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"VLMAnalyse erfolgreich: {filename}")
                    
                    # Kommentar
                    standardized_result = self._standardize_result(result, filename, processing_time)
                    return standardized_result
                    
                else:
                    error_text = await response.text()
                    logger.warning(f"APIFehler {response.status}: {error_text[:200]}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"Fehler bei der Verarbeitung{filename} -> {endpoint}")
            return None
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung{filename} -> {endpoint}: {e}")
            return None
    
    def _standardize_result(self, raw_result: Any, filename: str, processing_time: float) -> Dict[str, Any]:
        """
        Hinweis
        
        Args:
            raw_result: Hinweis
            filename: Dateiname
            processing_time: Hinweis
            
        Returns:
            Hinweis
        """
        if not raw_result:
            return {"result": [], "metadata": {"error": "Empty result"}}
        
        # Kommentar
        if isinstance(raw_result, dict) and "result" in raw_result:
            return raw_result
        
        # Kommentar
        content_list = []
        
        if isinstance(raw_result, list):
            content_list = raw_result
        elif isinstance(raw_result, dict):
            # Kommentar
            for key in ["content", "result", "data", "content_list", "parsed_content"]:
                if key in raw_result:
                    content_candidate = raw_result[key]
                    if isinstance(content_candidate, list):
                        content_list = content_candidate
                        break
        
        # Kommentar
        standardized = {
            "result": content_list,
            "metadata": {
                "filename": filename,
                "processing_time": processing_time,
                "mode": "vlm",
                "backend": "sglang-remote",
                "content_blocks": len(content_list),
                "raw_result_type": type(raw_result).__name__
            }
        }
        
        # Kommentar
        if isinstance(raw_result, dict):
            if "errors" in raw_result:
                standardized["metadata"]["errors"] = raw_result["errors"]
            if "warnings" in raw_result:
                standardized["metadata"]["warnings"] = raw_result["warnings"]
        
        return standardized
    
    async def process_document_bytes(self, file_bytes: bytes, filename: str, strategy: str = "vlm") -> Optional[Dict[str, Any]]:
        """
        Hinweis
        Hinweis
        
        Args:
            file_bytes: PDFDateibytes
            filename: Dateiname
            strategy: Hinweis
            
        Returns:
            Hinweis
        """
        if not file_bytes:
            logger.error(f"Dateidaten sind leer: {filename}")
            return None
        
        file_size_mb = len(file_bytes) / 1024 / 1024
        logger.info(f"Hinweis{filename} ({file_size_mb:.2f} MB)")
        
        # Kommentar
        is_healthy, server_info = await self.check_server_health()
        if not is_healthy:
            logger.error(f"Fehler bei der Verarbeitung{filename}")
            return None
        
        async with await self._create_session() as session:
            # Kommentar
            endpoint = await self._discover_parse_endpoint(session)
            if not endpoint:
                logger.error(f"Fehler bei der Verarbeitung{filename}")
                return None
            
            # Kommentar
            main_request = self._prepare_request_data(file_bytes, filename)
            
            # Kommentar
            result = await self._parse_with_endpoint(session, endpoint, main_request, filename)
            if result:
                return result
            
            # Kommentar
            logger.info(f"Hinweis{filename}")
            alternative_formats = self._prepare_alternative_formats(file_bytes, filename)
            
            for i, alt_format in enumerate(alternative_formats, 1):
                logger.info(f"Hinweis{i}/{len(alternative_formats)}: {filename}")
                result = await self._parse_with_endpoint(session, endpoint, alt_format, filename)
                if result:
                    return result
            
            logger.error(f"Fehler bei der Verarbeitung{filename}")
            return None


# Kommentar
def create_optimized_vlm_processor() -> OptimizedMinerUVLMClient:
    """
    Hinweis
    Hinweis
    
    Returns:
        OptimizedMinerUVLMClientHinweis
    """
    try:
        return OptimizedMinerUVLMClient()
    except ValueError as e:
        logger.error(f"Fehler bei der Verarbeitung{e}")
        logger.error("Fehler bei der Verarbeitung")
        raise


# Kommentar
class OptimizedMinerUProcessor:
    """
    Hinweis
    Hinweis
    """
    
    def __init__(self):
        self.vlm_client = OptimizedMinerUVLMClient()
        logger.info("OptimizedMinerUProcessor Hinweis")
    
    async def process_document_bytes(self, file_bytes: bytes, filename: str, strategy: str = "vlm") -> Optional[Dict[str, Any]]:
        """
        Hinweis
        
        Args:
            file_bytes: Dateibytes
            filename: Dateiname 
            strategy: Hinweis
            
        Returns:
            Hinweis
        """
        logger.info(f"Hinweis{filename} (Hinweis{strategy})")
        return await self.vlm_client.process_document_bytes(file_bytes, filename, "vlm")


# Kommentar
_optimized_processor = None

def get_optimized_mineru_processor():
    """
    Hinweis
    Hinweis
    
    Returns:
        OptimizedMinerUProcessorHinweis
    """
    global _optimized_processor
    if _optimized_processor is None:
        _optimized_processor = OptimizedMinerUProcessor()
        logger.info("Hinweis")
    return _optimized_processor
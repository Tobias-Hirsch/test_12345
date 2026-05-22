"""
Hinweis
Hinweis

Hinweis
1. Hinweis
2. Hinweis
3. Hinweis
4. Hinweis

Hinweis
1. Hinweis
2. OderHinweis
3. Hinweis

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
    from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2
    PDF_PREPROCESSING_AVAILABLE = True
    logger.info("PDFHinweis")
except ImportError as e:
    logger.warning(f"PDFWarnhinweis{e}")
    PDF_PREPROCESSING_AVAILABLE = False
    
    # Kommentar
    def convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes: bytes) -> bytes:
        """Hinweis"""
        logger.debug("Hinweis")
        return pdf_bytes


class OptimizedVLMProcessor:
    """
    Hinweis
    Hinweis
    """
    
    def __init__(self):
        """Hinweis"""
        self.server_url = settings.MINERU_SGLANG_SERVER_URL
        
        if not self.server_url:
            logger.error("MINERU_SGLANG_SERVER_URL Nicht konfiguriert")
            logger.error("Fehler bei der Verarbeitung=http://1.116.119.85:8908")
            self.server_url = None
            return
        
        # Kommentar
        parsed_url = urllib.parse.urlparse(self.server_url)
        self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Kommentar
        self.timeout_seconds = 600  # 10Hinweis
        self.max_retries = 3
        self.retry_delay = 2.0
        
        logger.info(f"OptimizedVLMProcessor Hinweis")
        logger.info(f"Hinweis{self.base_url}")
        logger.info(f"Hinweis{self.timeout_seconds}Hinweis")
        logger.info(f"Hinweis{self.max_retries}Hinweis")
    
    def _create_session(self) -> aiohttp.ClientSession:
        """Hinweis"""
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=5,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "RostiAI-MinerU-VLM/1.0",
            "Accept": "application/json"
        }
        
        return aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=headers
        )
    
    async def _check_server_health(self) -> bool:
        """Hinweis"""
        if not self.server_url:
            return False
        
        health_endpoints = ["/health", "/api/health", "/api/v1/health", "/status", "/"]
        
        try:
            async with self._create_session() as session:
                for endpoint in health_endpoints:
                    try:
                        async with session.get(f"{self.base_url}{endpoint}", timeout=aiohttp.ClientTimeout(total=10)) as response:
                            if response.status == 200:
                                logger.debug(f"Server-Healthcheck erfolgreich: {endpoint}")
                                return True
                    except:
                        continue
        except Exception as e:
            logger.error(f"Server-Healthcheck fehlgeschlagen: {e}")
        
        return False
    
    def _prepare_vlm_request(self, pdf_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Hinweis
        Hinweis
        """
        pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')
        
        # Kommentar
        primary_format = {
            "endpoint": "/api/v1/parse_pdf",
            "data": {
                "file_data": pdf_b64,
                "filename": filename,
                "mode": "vlm",
                "backend": "sglang",
                "config": {
                    "formula_enable": True,
                    "table_enable": True,
                    "parse_method": "auto",
                    "lang": "auto"
                }
            }
        }
        
        # Kommentar
        alternative_formats = [
            {
                "endpoint": "/api/v1/vlm/parse",
                "data": {
                    "pdf_data": pdf_b64,
                    "filename": filename,
                    "options": {
                        "parse_formulas": True,
                        "parse_tables": True,
                        "language": "auto"
                    }
                }
            },
            {
                "endpoint": "/api/parse",
                "data": {
                    "document": pdf_b64,
                    "name": filename,
                    "type": "pdf",
                    "backend": "vlm-sglang"
                }
            },
            {
                "endpoint": "/parse",
                "data": {
                    "file": pdf_b64,
                    "filename": filename,
                    "parser": "mineru-vlm"
                }
            }
        ]
        
        return [primary_format] + alternative_formats
    
    async def _vlm_parse_request(
        self, 
        session: aiohttp.ClientSession, 
        endpoint: str, 
        request_data: Dict[str, Any],
        filename: str
    ) -> Optional[Dict[str, Any]]:
        """führt ausHinweis"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            logger.debug(f"VLMHinweis{endpoint}")
            start_time = time.time()
            
            async with session.post(url, json=request_data) as response:
                processing_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"VLMAnalyse erfolgreich: {filename} (Dauer: {processing_time:.2f}s)")
                    
                    # Kommentar
                    return self._standardize_result(result, filename, processing_time)
                
                elif response.status == 404:
                    logger.debug(f"Hinweis{endpoint}")
                    return None
                
                else:
                    error_text = await response.text()
                    logger.warning(f"VLMAnfrage fehlgeschlagen {response.status}: {error_text[:200]}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"VLMZeitüberschreitung bei Anfrage: {endpoint}")
            return None
        except Exception as e:
            logger.error(f"VLMAnfragefehler: {endpoint} - {e}")
            return None
    
    def _standardize_result(self, raw_result: Any, filename: str, processing_time: float) -> Dict[str, Any]:
        """
        Hinweis
        
        Hinweis
        {
            "result": [...] # Hinweis
        }
        """
        if not raw_result:
            logger.warning(f"VLMWarnhinweis{filename}")
            return {"result": []}
        
        # Kommentar
        if isinstance(raw_result, dict) and "result" in raw_result:
            logger.debug(f"Ergebnisse{filename}")
            return raw_result
        
        # Kommentar
        content_list = []
        
        if isinstance(raw_result, list):
            content_list = raw_result
            logger.debug(f"ErgebnisseJaHinweis{len(content_list)} Einträge")
        
        elif isinstance(raw_result, dict):
            # Kommentar
            content_keys = ["content", "result", "data", "content_list", "parsed_content", "blocks"]
            
            for key in content_keys:
                if key in raw_result and isinstance(raw_result[key], list):
                    content_list = raw_result[key]
                    logger.debug(f"Hinweis'{key}' Hinweis{len(content_list)} Einträge")
                    break
        
        # Kommentar
        result = {"result": content_list}
        
        # Kommentar
        if isinstance(raw_result, dict):
            metadata = {
                "filename": filename,
                "processing_time": processing_time,
                "mode": "vlm-optimized",
                "content_blocks": len(content_list)
            }
            
            # Kommentar
            if "errors" in raw_result:
                metadata["errors"] = raw_result["errors"]
            if "warnings" in raw_result:
                metadata["warnings"] = raw_result["warnings"]
            
            result["_metadata"] = metadata
        
        logger.info(f"Hinweis{filename} - {len(content_list)} Hinweis")
        return result
    
    async def process_document_bytes(self, file_bytes: bytes, filename: str, strategy: str = "vlm") -> Optional[Dict[str, Any]]:
        """
        Hinweis
        
        Args:
            file_bytes: Dateibytes
            filename: Dateiname
            strategy: Hinweis
            
        Returns:
            Hinweis{"result": [...]} Oder None
        """
        if not file_bytes:
            logger.error(f"Dateidaten sind leer: {filename}")
            return None
        
        if not self.server_url:
            logger.error(f"Fehler bei der Verarbeitung{filename}")
            return None
        
        file_size_mb = len(file_bytes) / 1024 / 1024
        logger.info(f"Hinweis{filename} ({file_size_mb:.2f} MB)")
        
        # PDFKommentar
        if filename.lower().endswith('.pdf') and PDF_PREPROCESSING_AVAILABLE:
            try:
                logger.debug(f"Hinweis{filename}")
                file_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(file_bytes)
                logger.debug(f"PDFHinweis{filename}")
            except Exception as e:
                logger.warning(f"PDFWarnhinweis{filename} - {e}")
        
        # Kommentar
        if not await self._check_server_health():
            logger.error(f"Fehler bei der Verarbeitung{filename}")
            return None
        
        # Kommentar
        request_formats = self._prepare_vlm_request(file_bytes, filename)
        
        # Kommentar
        async with self._create_session() as session:
            for i, format_config in enumerate(request_formats, 1):
                logger.debug(f"Hinweis{i}/{len(request_formats)}: {format_config['endpoint']}")
                
                result = await self._vlm_parse_request(
                    session, 
                    format_config["endpoint"], 
                    format_config["data"],
                    filename
                )
                
                if result:
                    logger.info(f"VLMVerarbeitung erfolgreich: {filename} (Hinweis{i})")
                    return result
                
                # Kommentar
                if i < len(request_formats):
                    await asyncio.sleep(0.5)
        
        logger.error(f"Fehler bei der Verarbeitung{filename}")
        return None


class LocalMinerUParser:
    """
    Hinweis
    Hinweis
    """
    
    def __init__(self):
        logger.warning("LocalMinerUParser Warnhinweis")
        logger.warning("Warnhinweis")
    
    async def process_document_bytes(self, file_bytes: bytes, filename: str, strategy: str = "pipeline") -> Optional[Dict[str, Any]]:
        """Hinweis"""
        logger.error(f"Fehler bei der Verarbeitung{filename}")
        logger.error("Fehler bei der Verarbeitung")
        return None


class MinerUClient:
    """
    Hinweis
    Hinweis
    """
    
    def __init__(self):
        self.optimized_processor = OptimizedVLMProcessor()
        logger.info("MinerUClient Hinweis")
    
    async def process_document_bytes(self, file_bytes: bytes, filename: str, strategy: str) -> Optional[Dict[str, Any]]:
        """Hinweis"""
        return await self.optimized_processor.process_document_bytes(file_bytes, filename, "vlm")


# Kommentar
_mineru_processor = None

def get_mineru_processor():
    """
    Hinweis
    Hinweis
    """
    global _mineru_processor
    
    if _mineru_processor is None:
        # Kommentar
        if settings.MINERU_SGLANG_SERVER_URL:
            logger.info("Hinweis")
            _mineru_processor = OptimizedVLMProcessor()
        else:
            logger.warning("Nicht konfiguriertWarnhinweis")
            logger.warning("Warnhinweis=http://1.116.119.85:8908")
            _mineru_processor = LocalMinerUParser()
    
    return _mineru_processor


# Kommentar
mineru_client = get_mineru_processor()

# Kommentar
__all__ = [
    "get_mineru_processor",
    "mineru_client", 
    "OptimizedVLMProcessor",
    "LocalMinerUParser",
    "MinerUClient"
]
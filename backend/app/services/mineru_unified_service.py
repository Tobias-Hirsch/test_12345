"""
Hinweis
Hinweis

Hinweis
1. Hinweis
2. Hinweis
3. Hinweis
4. Hinweis
5. Hinweis

Hinweis
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
from datetime import datetime

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
    
    def convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes: bytes) -> bytes:
        """Hinweis"""
        logger.debug("Hinweis")
        return pdf_bytes


class ProcessingStrategy:
    """Hinweis"""
    
    def __init__(self, name: str):
        self.name = name
        self.success_count = 0
        self.failure_count = 0
        self.total_processing_time = 0.0
    
    async def process(self, file_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """Hinweis"""
        raise NotImplementedError("Fehler bei der Verarbeitung")
    
    def get_success_rate(self) -> float:
        """Hinweis"""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    def get_average_processing_time(self) -> float:
        """Hinweis"""
        return self.total_processing_time / self.success_count if self.success_count > 0 else 0.0
    
    def record_success(self, processing_time: float):
        """Hinweis"""
        self.success_count += 1
        self.total_processing_time += processing_time
    
    def record_failure(self):
        """Hinweis"""
        self.failure_count += 1


class SGLangStrategy(ProcessingStrategy):
    """SGLangHinweis"""
    
    def __init__(self):
        super().__init__("sglang")
        self.server_url = settings.MINERU_SGLANG_SERVER_URL
        self.timeout_seconds = 600
        self.max_retries = 3
        
        if self.server_url:
            parsed_url = urllib.parse.urlparse(self.server_url)
            self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        else:
            self.base_url = None
    
    async def process(self, file_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """SGLangHinweis"""
        if not self.base_url:
            logger.error("SGLangFehler bei der Verarbeitung")
            return None
        
        start_time = time.time()
        
        try:
            # Kommentar
            if not await self._check_server_health():
                logger.error("SGLangFehler bei der Verarbeitung")
                self.record_failure()
                return None
            
            # Kommentar
            pdf_b64 = base64.b64encode(file_bytes).decode('utf-8')
            request_data = {
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
            
            # führt ausKommentar
            async with self._create_session() as session:
                async with session.post(f"{self.base_url}/api/v1/parse_pdf", json=request_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        processing_time = time.time() - start_time
                        self.record_success(processing_time)
                        
                        logger.info(f"SGLangVerarbeitung erfolgreich: {filename} (Dauer: {processing_time:.2f}s)")
                        return self._standardize_result(result, filename)
                    else:
                        error_text = await response.text()
                        logger.error(f"SGLangFehler bei der Verarbeitung{response.status}: {error_text[:200]}")
                        self.record_failure()
                        return None
        
        except Exception as e:
            logger.error(f"SGLangAusnahme bei Verarbeitung: {filename} - {e}")
            self.record_failure()
            return None
    
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
            "User-Agent": "RostiAI-MinerU-Unified/1.0",
            "Accept": "application/json"
        }
        
        return aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=headers
        )
    
    async def _check_server_health(self) -> bool:
        """Hinweis"""
        health_endpoints = ["/health", "/api/health", "/api/v1/health", "/status", "/"]
        
        try:
            async with self._create_session() as session:
                for endpoint in health_endpoints:
                    try:
                        async with session.get(f"{self.base_url}{endpoint}", timeout=aiohttp.ClientTimeout(total=10)) as response:
                            if response.status == 200:
                                logger.debug(f"SGLangServer-Healthcheck erfolgreich: {endpoint}")
                                return True
                    except:
                        continue
        except Exception as e:
            logger.error(f"SGLangServer-Healthcheck fehlgeschlagen: {e}")
        
        return False
    
    def _standardize_result(self, raw_result: Any, filename: str) -> Dict[str, Any]:
        """Hinweis"""
        if not raw_result:
            return {"result": []}
        
        # Kommentar
        if isinstance(raw_result, dict) and "result" in raw_result:
            return raw_result
        
        # Kommentar
        content_list = []
        if isinstance(raw_result, list):
            content_list = raw_result
        elif isinstance(raw_result, dict):
            content_keys = ["content", "result", "data", "content_list", "parsed_content", "blocks"]
            for key in content_keys:
                if key in raw_result and isinstance(raw_result[key], list):
                    content_list = raw_result[key]
                    break
        
        return {"result": content_list}


class VLMStrategy(ProcessingStrategy):
    """Hinweis"""
    
    def __init__(self):
        super().__init__("vlm")
        # VLMKommentar
    
    async def process(self, file_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """Hinweis"""
        logger.info(f"VLMHinweis{filename}")
        self.record_failure()
        return None


class PipelineStrategy(ProcessingStrategy):
    """PipelineHinweis"""
    
    def __init__(self):
        super().__init__("pipeline")
    
    async def process(self, file_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """PipelineHinweis"""
        logger.info(f"PipelineHinweis{filename}")
        self.record_failure()
        return None


class FallbackStrategy(ProcessingStrategy):
    """Hinweis"""
    
    def __init__(self):
        super().__init__("fallback")
    
    async def process(self, file_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """PyMuPDFHinweis"""
        try:
            import fitz  # PyMuPDF
            
            start_time = time.time()
            
            # Kommentar
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text.strip():
                    full_text += f"\n\n--- Hinweis{page_num + 1}Hinweis\n{page_text}"
            
            doc.close()
            
            if full_text.strip():
                processing_time = time.time() - start_time
                self.record_success(processing_time)
                
                # Kommentar
                result = {
                    "result": [
                        {
                            "type": "text",
                            "text": full_text.strip(),
                            "page_idx": 0
                        }
                    ]
                }
                
                logger.info(f"FallbackVerarbeitung erfolgreich: {filename} (Dauer: {processing_time:.2f}s)")
                return result
            else:
                logger.warning(f"FallbackWarnhinweis{filename}")
                self.record_failure()
                return None
                
        except Exception as e:
            logger.error(f"FallbackVerarbeitung fehlgeschlagen: {filename} - {e}")
            self.record_failure()
            return None


class MinerUConfig:
    """MinerUHinweis"""
    
    @classmethod
    def get_processing_strategy(cls) -> str:
        """Hinweis"""
        # Kommentar
        if settings.MINERU_FORCE_MODE:
            return settings.MINERU_FORCE_MODE
        
        # Kommentar
        environment = settings.ENVIRONMENT.lower()
        current_hour = datetime.now().hour
        
        # Kommentar
        night_hours = settings.MINERU_NIGHTTIME_HOURS.split("-")
        if len(night_hours) == 2:
            night_start = int(night_hours[0])
            night_end = int(night_hours[1])
            is_nighttime = (
                (night_start > night_end and (current_hour >= night_start or current_hour < night_end)) or
                (night_start < night_end and night_start <= current_hour < night_end)
            )
        else:
            is_nighttime = False
        
        # Kommentar
        if environment == "production" and is_nighttime:
            return "sglang"  # Hinweis
        elif environment == "production":
            return "fallback"  # Hinweis
        elif environment == "development":
            return "sglang" if settings.MINERU_SGLANG_SERVER_URL else "fallback"
        else:
            return "fallback"  # Hinweis
    
    @classmethod
    def validate_configuration(cls) -> Dict[str, bool]:
        """Hinweisäge"""
        validations = {
            "sglang_server_configured": bool(settings.MINERU_SGLANG_SERVER_URL),
            "environment_set": bool(settings.ENVIRONMENT),
            "nighttime_hours_valid": cls._validate_nighttime_hours(),
            "pdf_preprocessing_available": PDF_PREPROCESSING_AVAILABLE
        }
        
        return validations
    
    @classmethod
    def _validate_nighttime_hours(cls) -> bool:
        """Hinweis"""
        try:
            hours = settings.MINERU_NIGHTTIME_HOURS.split("-")
            if len(hours) != 2:
                return False
            start, end = int(hours[0]), int(hours[1])
            return 0 <= start <= 23 and 0 <= end <= 23
        except:
            return False


class MinerUErrorHandler:
    """MinerUFehlerhinweis"""
    
    def __init__(self):
        self.error_counts = {}
        self.performance_metrics = {}
    
    async def with_retry(self, func, filename: str, max_retries: int = 3, *args, **kwargs):
        """Hinweis"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                result = await func(*args, **kwargs)
                if result:
                    return result
                else:
                    logger.warning(f"Warnhinweis{attempt + 1}/{max_retries} Warnhinweis{filename} (ErgebnisseWarnhinweis")
            except Exception as e:
                last_exception = e
                logger.warning(f"Warnhinweis{attempt + 1}/{max_retries} Warnhinweis{filename} - {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Hinweis
        
        # Kommentar
        self._record_error(filename, last_exception or Exception("Fehler bei der Verarbeitung"))
        return None
    
    def _record_error(self, filename: str, error: Exception):
        """Hinweis"""
        error_type = type(error).__name__
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        logger.error(f"Fehler bei der Verarbeitung{filename} - {error_type}: {error}")
    
    def log_processing_metrics(self, filename: str, strategy: str, duration: float, success: bool):
        """Hinweis"""
        if strategy not in self.performance_metrics:
            self.performance_metrics[strategy] = {
                "success_count": 0,
                "failure_count": 0,
                "total_time": 0.0,
                "files_processed": []
            }
        
        metrics = self.performance_metrics[strategy]
        if success:
            metrics["success_count"] += 1
            metrics["total_time"] += duration
        else:
            metrics["failure_count"] += 1
        
        metrics["files_processed"].append({
            "filename": filename,
            "duration": duration,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
        
        # Kommentar
        if len(metrics["files_processed"]) > 100:
            metrics["files_processed"] = metrics["files_processed"][-100:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Hinweis"""
        summary = {}
        for strategy, metrics in self.performance_metrics.items():
            total = metrics["success_count"] + metrics["failure_count"]
            avg_time = metrics["total_time"] / metrics["success_count"] if metrics["success_count"] > 0 else 0
            
            summary[strategy] = {
                "success_rate": metrics["success_count"] / total if total > 0 else 0,
                "average_processing_time": avg_time,
                "total_processed": total
            }
        
        return summary


class UnifiedMinerUProcessor:
    """Hinweis"""
    
    def __init__(self):
        """Hinweis"""
        self.strategies = {
            'sglang': SGLangStrategy(),
            'vlm': VLMStrategy(),
            'pipeline': PipelineStrategy(),
            'fallback': FallbackStrategy()
        }
        
        self.config = MinerUConfig()
        self.error_handler = MinerUErrorHandler()
        
        # Kommentar
        config_validation = self.config.validate_configuration()
        logger.info(f"MinerUHinweis{config_validation}")
        
        # Kommentar
        self._report_configuration_status()
    
    def _report_configuration_status(self):
        """Hinweis"""
        strategy = self.config.get_processing_strategy()
        logger.info(f"UnifiedMinerUProcessorHinweis")
        logger.info(f"Hinweis{strategy}")
        logger.info(f"Hinweis{settings.ENVIRONMENT}")
        logger.info(f"Hinweis{settings.MINERU_NIGHTTIME_HOURS}")
        
        if strategy == "sglang" and settings.MINERU_SGLANG_SERVER_URL:
            logger.info(f"SGLangHinweis{settings.MINERU_SGLANG_SERVER_URL}")
        elif strategy == "sglang":
            logger.warning("Warnhinweis")
    
    async def process_document_bytes(self, file_bytes: bytes, filename: str, strategy: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Hinweis
        
        Args:
            file_bytes: Dateibytes
            filename: Dateiname
            strategy: Hinweis
        
        Returns:
            Hinweis
        """
        if not file_bytes:
            logger.error(f"Dateidaten sind leer: {filename}")
            return None
        
        # Kommentar
        selected_strategy = strategy or self.config.get_processing_strategy()
        
        # PDFKommentar
        if filename.lower().endswith('.pdf') and PDF_PREPROCESSING_AVAILABLE:
            try:
                logger.debug(f"Hinweis{filename}")
                file_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(file_bytes)
            except Exception as e:
                logger.warning(f"PDFWarnhinweis{filename} - {e}")
        
        file_size_mb = len(file_bytes) / 1024 / 1024
        logger.info(f"Hinweis{filename} ({file_size_mb:.2f} MB) - Hinweis{selected_strategy}")
        
        start_time = time.time()
        
        # Kommentar
        result = await self._try_strategy(selected_strategy, file_bytes, filename)
        
        # Kommentar
        if not result and selected_strategy != "fallback":
            logger.warning(f"Warnhinweis{selected_strategy} Warnhinweis")
            result = await self._try_strategy("fallback", file_bytes, filename)
        
        # Kommentar
        processing_time = time.time() - start_time
        success = result is not None
        self.error_handler.log_processing_metrics(filename, selected_strategy, processing_time, success)
        
        if result:
            logger.info(f"Verarbeitung erfolgreich: {filename} (Dauer: {processing_time:.2f}s)")
        else:
            logger.error(f"Fehler bei der Verarbeitung{filename}")
        
        return result
    
    async def _try_strategy(self, strategy_name: str, file_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """Hinweis"""
        if strategy_name not in self.strategies:
            logger.error(f"Fehler bei der Verarbeitung{strategy_name}")
            return None
        
        strategy = self.strategies[strategy_name]
        
        # Kommentar
        result = await self.error_handler.with_retry(
            strategy.process,
            filename,
            3,
            file_bytes,
            filename
        )
        
        return result
    
    def get_strategy_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Hinweis"""
        stats = {}
        for name, strategy in self.strategies.items():
            stats[name] = {
                "success_count": strategy.success_count,
                "failure_count": strategy.failure_count,
                "success_rate": strategy.get_success_rate(),
                "average_processing_time": strategy.get_average_processing_time()
            }
        
        return stats
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Hinweis"""
        return {
            "strategy_stats": self.get_strategy_statistics(),
            "error_handler_metrics": self.error_handler.get_performance_summary(),
            "configuration": self.config.validate_configuration()
        }


# Kommentar
_unified_processor = None

def get_unified_mineru_processor() -> UnifiedMinerUProcessor:
    """Hinweis"""
    global _unified_processor
    
    if _unified_processor is None:
        _unified_processor = UnifiedMinerUProcessor()
    
    return _unified_processor


# Kommentar
def get_mineru_processor():
    """Hinweis"""
    return get_unified_mineru_processor()


# Kommentar
__all__ = [
    "UnifiedMinerUProcessor",
    "get_unified_mineru_processor", 
    "get_mineru_processor",
    "MinerUConfig",
    "MinerUErrorHandler"
]

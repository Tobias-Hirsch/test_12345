"""
MinerUHinweis
Hinweis

Hinweis
1. Hinweis
2. Hinweis
3. Hinweis
4. Hinweis

Hinweis
Hinweis
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Hinweis"""
    strategy: str
    max_retries: int
    timeout_seconds: int
    enable_preprocessing: bool
    fallback_enabled: bool


@dataclass
class ServerConfig:
    """Hinweis"""
    sglang_url: Optional[str]
    health_check_timeout: int
    connection_pool_size: int
    request_timeout: int


@dataclass
class PerformanceConfig:
    """Hinweis"""
    max_concurrent_jobs: int
    memory_limit_mb: int
    cache_enabled: bool
    metrics_enabled: bool


class MinerUConfigManager:
    """MinerUHinweis"""
    
    def __init__(self):
        """Hinweis"""
        self._processing_config = None
        self._server_config = None
        self._performance_config = None
        self._config_cache = {}
        self._last_validation = None
        
        # Kommentar
        self._initialize_configs()
        
        logger.info("MinerUHinweis")
    
    def _initialize_configs(self):
        """Hinweis"""
        self._processing_config = self._build_processing_config()
        self._server_config = self._build_server_config()
        self._performance_config = self._build_performance_config()
    
    def _build_processing_config(self) -> ProcessingConfig:
        """Hinweis"""
        return ProcessingConfig(
            strategy=self.get_optimal_strategy(),
            max_retries=getattr(settings, 'MINERU_MAX_RETRIES', 3),
            timeout_seconds=getattr(settings, 'MINERU_TIMEOUT_SECONDS', 600),
            enable_preprocessing=getattr(settings, 'MINERU_ENABLE_PREPROCESSING', True),
            fallback_enabled=getattr(settings, 'MINERU_FALLBACK_ENABLED', True)
        )
    
    def _build_server_config(self) -> ServerConfig:
        """Hinweis"""
        return ServerConfig(
            sglang_url=self._validate_and_normalize_url(settings.MINERU_SGLANG_SERVER_URL),
            health_check_timeout=getattr(settings, 'MINERU_HEALTH_CHECK_TIMEOUT', 10),
            connection_pool_size=getattr(settings, 'MINERU_CONNECTION_POOL_SIZE', 10),
            request_timeout=getattr(settings, 'MINERU_REQUEST_TIMEOUT', 600)
        )
    
    def _build_performance_config(self) -> PerformanceConfig:
        """Hinweis"""
        return PerformanceConfig(
            max_concurrent_jobs=getattr(settings, 'MINERU_MAX_CONCURRENT_JOBS', 3),
            memory_limit_mb=getattr(settings, 'MINERU_MEMORY_LIMIT_MB', 2048),
            cache_enabled=getattr(settings, 'MINERU_CACHE_ENABLED', True),
            metrics_enabled=getattr(settings, 'MINERU_METRICS_ENABLED', True)
        )
    
    def get_optimal_strategy(self) -> str:
        """Hinweis"""
        # Kommentar
        force_mode = getattr(settings, 'MINERU_FORCE_MODE', '').lower()
        if force_mode in ['sglang', 'vlm', 'pipeline', 'fallback']:
            logger.info(f"Hinweis{force_mode}")
            return force_mode
        
        # Kommentar
        environment = settings.ENVIRONMENT.lower()
        current_hour = datetime.now().hour
        
        # Kommentar
        is_nighttime = self._is_nighttime(current_hour)
        
        # Kommentar
        sglang_available = bool(settings.MINERU_SGLANG_SERVER_URL)
        
        # Kommentar
        if environment == "production":
            if is_nighttime and sglang_available:
                strategy = "sglang"
                reason = "Hinweis"
            else:
                strategy = "fallback"
                reason = "Hinweis"
        elif environment == "development":
            if sglang_available:
                strategy = "sglang"
                reason = "Hinweis"
            else:
                strategy = "fallback"
                reason = "Hinweis"
        elif environment == "test":
            strategy = "fallback"
            reason = "Hinweis"
        else:
            strategy = "fallback"
            reason = "Hinweis"
        
        logger.info(f"Hinweis{strategy} ({reason})")
        return strategy
    
    def _is_nighttime(self, current_hour: int) -> bool:
        """Hinweis"""
        try:
            night_hours = getattr(settings, 'MINERU_NIGHTTIME_HOURS', '22-6')
            hours = night_hours.split('-')
            
            if len(hours) != 2:
                logger.warning(f"Warnhinweis{night_hours}, Warnhinweis")
                return False
            
            night_start = int(hours[0])
            night_end = int(hours[1])
            
            # Kommentar
            if not (0 <= night_start <= 23 and 0 <= night_end <= 23):
                logger.warning(f"Warnhinweis{night_hours}")
                return False
            
            # Kommentar
            if night_start > night_end:  # Hinweis
                return current_hour >= night_start or current_hour < night_end
            else:  # Hinweis
                return night_start <= current_hour < night_end
                
        except (ValueError, AttributeError) as e:
            logger.error(f"Fehler bei der Verarbeitung{e}")
            return False
    
    def _validate_and_normalize_url(self, url: Optional[str]) -> Optional[str]:
        """Hinweis"""
        if not url:
            return None
        
        # Kommentar
        if not url.startswith(('http://', 'https://')):
            logger.warning(f"URLWarnhinweis{url}")
            url = f"http://{url}"
        
        # Kommentar
        url = url.rstrip('/')
        
        return url
    
    def validate_all_configurations(self) -> Dict[str, bool]:
        """Hinweisäge"""
        validations = {}
        
        # Kommentar
        validations['environment_valid'] = settings.ENVIRONMENT.lower() in ['production', 'development', 'test']
        validations['nighttime_hours_valid'] = self._validate_nighttime_hours()
        
        # Kommentar
        validations['sglang_url_valid'] = self._validate_sglang_url()
        
        # Kommentar
        validations['strategy_valid'] = self._processing_config.strategy in ['sglang', 'vlm', 'pipeline', 'fallback']
        validations['retries_valid'] = 1 <= self._processing_config.max_retries <= 10
        validations['timeout_valid'] = 30 <= self._processing_config.timeout_seconds <= 1800
        
        # Kommentar
        validations['concurrent_jobs_valid'] = 1 <= self._performance_config.max_concurrent_jobs <= 20
        validations['memory_limit_valid'] = 512 <= self._performance_config.memory_limit_mb <= 8192
        
        # KommentarägeKommentar
        try:
            from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2
            validations['pdf_preprocessing_available'] = True
        except ImportError:
            validations['pdf_preprocessing_available'] = False
        
        # Kommentar
        self._last_validation = validations
        
        # Kommentar
        failed_validations = [k for k, v in validations.items() if not v]
        if failed_validations:
            logger.warning(f"WarnhinweisägeWarnhinweis{failed_validations}")
        else:
            logger.info("Hinweis")
        
        return validations
    
    def _validate_nighttime_hours(self) -> bool:
        """Hinweis"""
        try:
            night_hours = getattr(settings, 'MINERU_NIGHTTIME_HOURS', '22-6')
            hours = night_hours.split('-')
            
            if len(hours) != 2:
                return False
            
            start, end = int(hours[0]), int(hours[1])
            return 0 <= start <= 23 and 0 <= end <= 23
            
        except (ValueError, AttributeError):
            return False
    
    def _validate_sglang_url(self) -> bool:
        """Hinweis"""
        url = self._server_config.sglang_url
        if not url:
            return True  # Hinweis
        
        return url.startswith(('http://', 'https://')) and '://' in url
    
    def get_processing_config(self) -> ProcessingConfig:
        """Hinweis"""
        return self._processing_config
    
    def get_server_config(self) -> ServerConfig:
        """Hinweis"""
        return self._server_config
    
    def get_performance_config(self) -> PerformanceConfig:
        """Hinweis"""
        return self._performance_config
    
    def reload_configuration(self):
        """Hinweis"""
        logger.info("Hinweis")
        self._config_cache.clear()
        self._initialize_configs()
        self.validate_all_configurations()
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Hinweis"""
        return {
            "processing": {
                "strategy": self._processing_config.strategy,
                "max_retries": self._processing_config.max_retries,
                "timeout_seconds": self._processing_config.timeout_seconds,
                "preprocessing_enabled": self._processing_config.enable_preprocessing,
                "fallback_enabled": self._processing_config.fallback_enabled
            },
            "server": {
                "sglang_configured": bool(self._server_config.sglang_url),
                "sglang_url": self._server_config.sglang_url,
                "health_check_timeout": self._server_config.health_check_timeout,
                "connection_pool_size": self._server_config.connection_pool_size
            },
            "performance": {
                "max_concurrent_jobs": self._performance_config.max_concurrent_jobs,
                "memory_limit_mb": self._performance_config.memory_limit_mb,
                "cache_enabled": self._performance_config.cache_enabled,
                "metrics_enabled": self._performance_config.metrics_enabled
            },
            "environment": {
                "environment": settings.ENVIRONMENT,
                "nighttime_hours": getattr(settings, 'MINERU_NIGHTTIME_HOURS', '22-6'),
                "force_mode": getattr(settings, 'MINERU_FORCE_MODE', ''),
                "current_hour": datetime.now().hour,
                "is_nighttime": self._is_nighttime(datetime.now().hour)
            },
            "validation": self._last_validation or {}
        }
    
    def get_strategy_for_environment(self, environment: str, hour: Optional[int] = None) -> str:
        """Hinweis"""
        if hour is None:
            hour = datetime.now().hour
        
        is_night = self._is_nighttime(hour)
        sglang_available = bool(self._server_config.sglang_url)
        
        env = environment.lower()
        
        if env == "production":
            return "sglang" if is_night and sglang_available else "fallback"
        elif env == "development":
            return "sglang" if sglang_available else "fallback"
        elif env == "test":
            return "fallback"
        else:
            return "fallback"


# Kommentar
_config_manager = None

def get_mineru_config_manager() -> MinerUConfigManager:
    """Hinweis"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = MinerUConfigManager()
    
    return _config_manager


# Kommentar
def get_optimal_strategy() -> str:
    """Hinweis"""
    return get_mineru_config_manager().get_optimal_strategy()


def validate_mineru_configuration() -> Dict[str, bool]:
    """Hinweis"""
    return get_mineru_config_manager().validate_all_configurations()


def get_mineru_configuration_summary() -> Dict[str, Any]:
    """Hinweis"""
    return get_mineru_config_manager().get_configuration_summary()


# Kommentar
__all__ = [
    "MinerUConfigManager",
    "ProcessingConfig",
    "ServerConfig", 
    "PerformanceConfig",
    "get_mineru_config_manager",
    "get_optimal_strategy",
    "validate_mineru_configuration",
    "get_mineru_configuration_summary"
]
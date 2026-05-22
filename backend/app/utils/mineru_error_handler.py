"""
MinerUHinweis
Hinweis

Hinweis
1. Hinweis
2. Hinweis
3. Hinweis
4. Fehlerhinweis

Hinweis
Hinweis
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json


logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Fehlerhinweis"""
    NETWORK_ERROR = "network"
    TIMEOUT_ERROR = "timeout"
    AUTHENTICATION_ERROR = "auth"
    RATE_LIMIT_ERROR = "rate_limit"
    SERVER_ERROR = "server"
    CLIENT_ERROR = "client"
    PROCESSING_ERROR = "processing"
    CONFIGURATION_ERROR = "config"
    UNKNOWN_ERROR = "unknown"


@dataclass
class ErrorMetrics:
    """Fehlerhinweis"""
    category: ErrorCategory
    count: int = 0
    first_occurrence: Optional[datetime] = None
    last_occurrence: Optional[datetime] = None
    total_retry_attempts: int = 0
    successful_retries: int = 0
    
    def record_error(self, retry_attempt: int = 0, retry_success: bool = False):
        """Hinweis"""
        self.count += 1
        now = datetime.now()
        
        if self.first_occurrence is None:
            self.first_occurrence = now
        self.last_occurrence = now
        
        if retry_attempt > 0:
            self.total_retry_attempts += 1
            if retry_success:
                self.successful_retries += 1


@dataclass
class ProcessingMetrics:
    """Hinweis"""
    strategy: str
    filename: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    success: bool = False
    error_category: Optional[ErrorCategory] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    file_size_mb: float = 0.0
    
    def mark_completed(self, success: bool, error_category: Optional[ErrorCategory] = None, error_message: Optional[str] = None):
        """Hinweis"""
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.success = success
        self.error_category = error_category
        self.error_message = error_message


class RetryStrategy:
    """Hinweis"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0, exponential_base: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    def get_delay(self, attempt: int) -> float:
        """Hinweis"""
        if attempt <= 0:
            return 0.0
        
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        return min(delay, self.max_delay)
    
    def should_retry(self, attempt: int, error_category: ErrorCategory) -> bool:
        """Hinweis"""
        if attempt >= self.max_retries:
            return False
        
        # Kommentar
        non_retryable_errors = {
            ErrorCategory.AUTHENTICATION_ERROR,
            ErrorCategory.CONFIGURATION_ERROR,
            ErrorCategory.CLIENT_ERROR  # 4xxFehlerFehler bei der Verarbeitung
        }
        
        return error_category not in non_retryable_errors


class MinerUErrorHandler:
    """MinerUFehlerhinweis"""
    
    def __init__(self):
        self.error_metrics: Dict[ErrorCategory, ErrorMetrics] = {}
        self.processing_history: List[ProcessingMetrics] = []
        self.retry_strategy = RetryStrategy()
        
        # Kommentar
        for category in ErrorCategory:
            self.error_metrics[category] = ErrorMetrics(category)
        
        logger.info("MinerUFehlerhinweis")
    
    def classify_error(self, error: Exception) -> ErrorCategory:
        """Fehlerhinweis"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Kommentar
        if any(keyword in error_type.lower() for keyword in ['connection', 'network', 'dns', 'socket']):
            return ErrorCategory.NETWORK_ERROR
        
        # Kommentar
        if 'timeout' in error_type.lower() or 'timeout' in error_message:
            return ErrorCategory.TIMEOUT_ERROR
        
        # HTTPStatusKommentar
        if hasattr(error, 'status') or hasattr(error, 'status_code'):
            status = getattr(error, 'status', None) or getattr(error, 'status_code', None)
            if status:
                if status == 401:
                    return ErrorCategory.AUTHENTICATION_ERROR
                elif status == 429:
                    return ErrorCategory.RATE_LIMIT_ERROR
                elif 500 <= status < 600:
                    return ErrorCategory.SERVER_ERROR
                elif 400 <= status < 500:
                    return ErrorCategory.CLIENT_ERROR
        
        # Kommentar
        if any(keyword in error_message for keyword in ['auth', 'unauthorized', 'forbidden', 'token']):
            return ErrorCategory.AUTHENTICATION_ERROR
        
        # Kommentar
        if any(keyword in error_message for keyword in ['config', 'setting', 'environment', 'variable']):
            return ErrorCategory.CONFIGURATION_ERROR
        
        # Kommentar
        if any(keyword in error_message for keyword in ['parse', 'process', 'convert', 'extract']):
            return ErrorCategory.PROCESSING_ERROR
        
        return ErrorCategory.UNKNOWN_ERROR
    
    async def with_retry(
        self,
        func: Callable,
        filename: str,
        strategy: str,
        file_size_mb: float = 0.0,
        *args,
        **kwargs
    ) -> Tuple[Any, ProcessingMetrics]:
        """
        Hinweisührt ausHinweis
        
        Args:
            func: Hinweisührt ausHinweis
            filename: Dateiname
            strategy: Hinweis
            file_size_mb: Dateigröße(MB)
            *args, **kwargs: Kommentar
            
        Returns:
            (Ergebnisse, Hinweis
        """
        metrics = ProcessingMetrics(
            strategy=strategy,
            filename=filename,
            start_time=datetime.now(),
            file_size_mb=file_size_mb
        )
        
        last_error = None
        last_error_category = None
        
        for attempt in range(self.retry_strategy.max_retries + 1):
            try:
                # Kommentar
                metrics.retry_count = attempt
                
                # führt ausKommentar
                result = await func(*args, **kwargs)
                
                if result:
                    # Kommentar
                    metrics.mark_completed(True)
                    self.processing_history.append(metrics)
                    
                    # Kommentar
                    if attempt > 0 and last_error_category:
                        self.error_metrics[last_error_category].record_error(attempt, True)
                    
                    logger.info(f"Verarbeitung erfolgreich: {filename} (Hinweis{attempt + 1}, Dauer: {metrics.duration:.2f}s)")
                    return result, metrics
                else:
                    # ErgebnisseKommentar
                    error_category = ErrorCategory.PROCESSING_ERROR
                    error_message = "Fehler bei der Verarbeitung"
                    
                    if attempt < self.retry_strategy.max_retries:
                        logger.warning(f"Warnhinweis{filename} (Warnhinweis{attempt + 1})")
                        last_error_category = error_category
                        
                        # Kommentar
                        delay = self.retry_strategy.get_delay(attempt + 1)
                        if delay > 0:
                            await asyncio.sleep(delay)
                        continue
                    else:
                        # Kommentar
                        metrics.mark_completed(False, error_category, error_message)
                        self.error_metrics[error_category].record_error(attempt)
                        self.processing_history.append(metrics)
                        
                        logger.error(f"Fehler bei der Verarbeitung{filename} (Fehler bei der Verarbeitung{error_message})")
                        return None, metrics
            
            except Exception as e:
                last_error = e
                error_category = self.classify_error(e)
                last_error_category = error_category
                
                # Kommentar
                if attempt < self.retry_strategy.max_retries and self.retry_strategy.should_retry(attempt, error_category):
                    logger.warning(f"Fehler bei der Verarbeitung{filename} (Fehler bei der Verarbeitung{attempt + 1}, Fehler: {error_category.value}) - {e}")
                    
                    # Kommentar
                    self.error_metrics[error_category].record_error(attempt, False)
                    
                    # Kommentar
                    delay = self.retry_strategy.get_delay(attempt + 1)
                    if delay > 0:
                        await asyncio.sleep(delay)
                    continue
                else:
                    # Kommentar
                    metrics.mark_completed(False, error_category, str(e))
                    self.error_metrics[error_category].record_error(attempt)
                    self.processing_history.append(metrics)
                    
                    logger.error(f"Fehler bei der Verarbeitung{filename} (Fehler: {error_category.value}) - {e}")
                    return None, metrics
        
        # Kommentarührt ausKommentar
        return None, metrics
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Hinweis"""
        stats = {}
        
        for category, metrics in self.error_metrics.items():
            if metrics.count > 0:
                retry_success_rate = (
                    metrics.successful_retries / metrics.total_retry_attempts 
                    if metrics.total_retry_attempts > 0 else 0
                )
                
                stats[category.value] = {
                    "count": metrics.count,
                    "first_occurrence": metrics.first_occurrence.isoformat() if metrics.first_occurrence else None,
                    "last_occurrence": metrics.last_occurrence.isoformat() if metrics.last_occurrence else None,
                    "total_retry_attempts": metrics.total_retry_attempts,
                    "successful_retries": metrics.successful_retries,
                    "retry_success_rate": retry_success_rate
                }
        
        return stats
    
    def get_processing_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Hinweis"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.processing_history if m.start_time >= cutoff_time]
        
        if not recent_metrics:
            return {"message": f"Hinweis{hours}Hinweis"}
        
        # Kommentar
        strategy_stats = {}
        for metrics in recent_metrics:
            strategy = metrics.strategy
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    "total_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "total_duration": 0.0,
                    "total_file_size_mb": 0.0,
                    "total_retries": 0
                }
            
            stats = strategy_stats[strategy]
            stats["total_count"] += 1
            stats["total_duration"] += metrics.duration
            stats["total_file_size_mb"] += metrics.file_size_mb
            stats["total_retries"] += metrics.retry_count
            
            if metrics.success:
                stats["success_count"] += 1
            else:
                stats["failure_count"] += 1
        
        # Kommentar
        for strategy, stats in strategy_stats.items():
            total = stats["total_count"]
            stats["success_rate"] = stats["success_count"] / total if total > 0 else 0
            stats["average_duration"] = stats["total_duration"] / total if total > 0 else 0
            stats["average_file_size_mb"] = stats["total_file_size_mb"] / total if total > 0 else 0
            stats["average_retries"] = stats["total_retries"] / total if total > 0 else 0
        
        # Kommentar
        total_files = len(recent_metrics)
        successful_files = sum(1 for m in recent_metrics if m.success)
        
        return {
            "time_range_hours": hours,
            "total_files_processed": total_files,
            "successful_files": successful_files,
            "failed_files": total_files - successful_files,
            "overall_success_rate": successful_files / total_files if total_files > 0 else 0,
            "strategy_breakdown": strategy_stats
        }
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Hinweis"""
        if not self.processing_history:
            return {"message": "Hinweis"}
        
        # Kommentar
        recent_metrics = self.processing_history[-100:]
        
        # Kommentar
        successful = [m for m in recent_metrics if m.success]
        failed = [m for m in recent_metrics if not m.success]
        
        insights = {
            "total_samples": len(recent_metrics),
            "success_rate": len(successful) / len(recent_metrics) if recent_metrics else 0
        }
        
        if successful:
            durations = [m.duration for m in successful]
            file_sizes = [m.file_size_mb for m in successful]
            
            insights["successful_processing"] = {
                "average_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "average_file_size_mb": sum(file_sizes) / len(file_sizes) if file_sizes else 0
            }
        
        if failed:
            # Kommentar
            failure_categories = {}
            for m in failed:
                category = m.error_category.value if m.error_category else "unknown"
                failure_categories[category] = failure_categories.get(category, 0) + 1
            
            insights["failure_analysis"] = {
                "common_failure_categories": failure_categories,
                "average_retries_before_failure": sum(m.retry_count for m in failed) / len(failed)
            }
        
        # Kommentar
        strategy_performance = {}
        for m in recent_metrics:
            if m.strategy not in strategy_performance:
                strategy_performance[m.strategy] = {"total": 0, "successful": 0}
            
            strategy_performance[m.strategy]["total"] += 1
            if m.success:
                strategy_performance[m.strategy]["successful"] += 1
        
        for strategy, perf in strategy_performance.items():
            perf["success_rate"] = perf["successful"] / perf["total"] if perf["total"] > 0 else 0
        
        insights["strategy_performance"] = strategy_performance
        
        return insights
    
    def clear_old_history(self, days: int = 7):
        """Hinweis"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        old_count = len(self.processing_history)
        self.processing_history = [m for m in self.processing_history if m.start_time >= cutoff_time]
        new_count = len(self.processing_history)
        
        logger.info(f"Hinweis{old_count - new_count} Hinweis{days} Hinweis")
    
    def export_metrics(self) -> Dict[str, Any]:
        """Hinweis"""
        return {
            "error_statistics": self.get_error_statistics(),
            "processing_statistics": self.get_processing_statistics(),
            "performance_insights": self.get_performance_insights(),
            "export_timestamp": datetime.now().isoformat()
        }


# Kommentar
_error_handler = None

def get_mineru_error_handler() -> MinerUErrorHandler:
    """Hinweis"""
    global _error_handler
    
    if _error_handler is None:
        _error_handler = MinerUErrorHandler()
    
    return _error_handler


# Kommentar
__all__ = [
    "MinerUErrorHandler",
    "ErrorCategory",
    "ErrorMetrics",
    "ProcessingMetrics",
    "RetryStrategy",
    "get_mineru_error_handler"
]
import json
import logging
from typing import AsyncGenerator, Any, Dict

logger = logging.getLogger(__name__)

async def sse_stream_formatter(generator: AsyncGenerator[Dict[str, Any], None]) -> AsyncGenerator[str, None]:
    """
    Formats the output of an asynchronous generator into Server-Sent Events (SSE).
    It accepts the event/data shape used by the normal chat endpoint and falls back
    to data-only SSE messages for older agent generators that yield plain objects.
    """
    try:
        async for chunk in generator:
            if isinstance(chunk, dict) and 'event' in chunk and 'data' in chunk:
                event = chunk['event']
                data = chunk['data']
                
                # ALWAYS JSON-encode the data part to handle special characters and ensure consistency
                data_str = json.dumps(data)
                
                yield f"event: {event}\ndata: {data_str}\n\n"
            else:
                logger.warning(f"sse_stream_formatter received chunk without event/data keys: {chunk}")
                yield f"data: {json.dumps(chunk)}\n\n"

    except Exception as e:
        logger.error(f"Error in sse_stream_formatter: {e}", exc_info=True)
        error_data = json.dumps({'error': str(e)})
        yield f"event: error\ndata: {error_data}\n\n"

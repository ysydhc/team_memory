import asyncio
import logging
from sqlalchemy import text
from team_memory.config import load_settings
from team_memory.storage.database import get_session
from team_memory.services.infra.task_runner import _handlers
import gc
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_tasks():
    settings = load_settings()
    db_url = str(settings.database.url)
    
    handler = _handlers.get('graphiti_ingest')
    if not handler:
        logger.error("graphiti_ingest handler not found")
        return
        
    processed_count = 0
    
    while True:
        async with get_session(db_url) as session:
            # Get one pending task
            res = await session.execute(
                text("SELECT id, payload FROM background_tasks WHERE task_type = 'graphiti_ingest' AND status = 'pending' ORDER BY created_at ASC LIMIT 1")
            )
            row = res.first()
            
            if not row:
                logger.info("No more pending tasks.")
                break
                
            task_id, payload = row
            
            logger.info(f"Processing task {task_id} (Record: {payload.get('record_id')})")
            
            try:
                # Mark as processing (started_at)
                await session.execute(
                    text("UPDATE background_tasks SET started_at = now() WHERE id = :id"),
                    {"id": task_id}
                )
                await session.commit()
                
                # Execute handler
                await handler(payload)
                
                # Mark as completed
                await session.execute(
                    text("UPDATE background_tasks SET status = 'completed', completed_at = now() WHERE id = :id"),
                    {"id": task_id}
                )
                await session.commit()
                logger.info(f"Task {task_id} completed successfully.")
                
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                await session.rollback()
                # Mark as failed
                await session.execute(
                    text("UPDATE background_tasks SET status = 'failed', error_message = :error WHERE id = :id"),
                    {"id": task_id, "error": str(e)}
                )
                await session.commit()
                
        processed_count += 1
        
        # Force garbage collection to free up memory
        gc.collect()
        
        # We need to restart the process periodically to avoid Kuzu memory leaks/locks
        if processed_count >= 10:
            logger.info("Processed 10 tasks, restarting to clear Kuzu state...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
            
        # Optional: Add a small sleep to avoid completely pegging the CPU
        await asyncio.sleep(0.5)

if __name__ == "__main__":
    import sys
    if sys.platform != "win32":
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    asyncio.run(process_tasks())


import uuid 
import logging
from client.client import A2AClient
from models.task import Task

logger = logging.getLogger(__name__)

class AgentConnector:
    
    def __init__(self, name:str, base_url:str):
        self.name = name
        self.client = A2AClient(url=base_url)
        logger.info(f"AgentConnector: initialized for {self.name} at {base_url}")

    async def send_task(self, message: str, session_id:str) -> Task:
        task_id = uuid.uuid4().hex
        payload = {
            "id": task_id,
            "sessionId": session_id,
            "message": {
                "role": "user",
                "parts": [
                    {"type":"text", "text":message}
                ]
            }
        }
        task_result = await self.client.send_task(payload)
        logger.info(f"AgentConnector: received respons from {self.name} for task {task_id}")
        return task_result
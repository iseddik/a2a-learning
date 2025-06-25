import os
import uuid
import logging
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents.llm_agent import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from server.task_manager import InMemoryTaskManager
from models.request import SendTaskRequest, SendTaskResponse
from models.task import Message, TaskStatus, TaskState, TexPart
from agents.host_agent.agent_connect import AgentConnector
from models.agent import AgentCard

logger = logging.getLogger(__name__)

class Orchestrator:
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]
    
    def __init__(self, agent_cards: list[AgentCard]):
        self.connectors = {
            card.name: AgentConnector(card.name, card.url) for card in agent_cards
        }
        self._agent = self._build_agent()
        self._user_id = "orchestrator_user"
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )
    
    def _build_agent(self) -> LlmAgent:
        return LlmAgent(
            model="gemini-1.5-flash-latest", # we can change here for a hf model if we want!
            name="orchestrator_agent",
            description="Delegates user queries to child A2A agents based on intent.",
            instruction=self._root_instruction,
            tools=[
                self._list_agents,
                self._delegate_task
            ],
        )
    
    def _root_instruction(self, context: ReadonlyContext) -> str:
        agent_list = "\n".join(f"- {name}" for name in self.connectors)
        return (
            "You are an orchestrator with two tools:\n"
            "1) list_agents() -> list available child agents\n"
            "2) delegate_task(agent_name, message)-> call that agent\n"
            "Use these tools to satisfy the user. Do not hallucinate.\n"
            "Available agents:\n" + agent_list
        )
    
    def _list_agents(self) -> list[str]:
        return list(self.connectors.keys())
    
    async def _delegate_task(self, agent_name: str, message:str, tool_context: ToolContext) -> str:
        if agent_name not in self.connectors:
            raise ValueError(f"Unknown agent: {agent_name}")
        connector = self.connectors[agent_name]

        state = tool_context.state
        if "session_id" not in state:
            state["session_id"] = str(uuid.uuid4())
        session_id = state["session_id"]

        child_task = await connector.send_task(message, session_id)

        if child_task.history and len(child_task.history) > 1:
            return child_task.history[-1].parts[0].text
        return ""
    
    def invoke(self, query: str, session_id:str) -> str:
        session = self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=session_id,
            state={}
        )
        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=query)]
        )
        events = list(
            self._runner.run(
                user_id=self._user_id,
                session_id=session.id,
                new_message=content
            )
        )
        if not events or not events[-1].content or not events[-1].content.parts:
            return ""
        return "\n".join(p.text for p in events[-1].content.parts if p.text)

class OrchestratorTaskManager(InMemoryTaskManager):

    def __init__(self, agent: Orchestrator):
        super().__init__()
        self.agent = agent
    
    def _get_user_text(self, request: SendTaskRequest) -> str:
        return request.params.message.parts[0].text
    
    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        logger.info(f"OrchestratorTaskManager received task {request.params.id}")
        task = await self.upsert_task(request.params)
        user_text = self._get_user_text(request=request)
        response_text = self.agent.invoke(user_text, request.params.sessionId)
        reply = Message(role="agent", parts=[TexPart(text=response_text)])
        async with self.lock:
            task.status = TaskStatus(state=TaskState.COMPLETED)
            task.history.append(reply)
        
        return SendTaskResponse(id=request.id, result=task)


    

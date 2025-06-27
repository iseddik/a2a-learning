import asyncio
import logging 
import click

from utils.discovery import Discover
from server.server import A2AServer
from models.agent import AgentCard, AgentCapabilities, AgentSkill
from agents.host_agent.orchestrator import (
    Orchestrator,
    OrchestratorTaskManager
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option(
    "--host", default="localhost",
    help="Host to bind the OrchestratorAgent server to"
)
@click.option(
    "--port", default=10002,
    help="Port for the OrchestratorAgent server"
)
@click.option(
    "--registry",
    default=None,
    help=(
        "Part to Json"
    )
) # i stoped here!
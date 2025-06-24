import os
import json
import logging
from typing import List

import httpx
from model.agent import AgentCard

logger = logging.getLogger(__name__)

class Discover:
    def __init__(self, regisrty_file: str = None):
        self.registry_file = regisrty_file # here we should give the path to all the listed remote agents 
        self.base_urls = self._load_registry()

    def _load_registry(self) -> List[str]:
        with open(self.registry_file, "r") as f:
            data = json.load(f)
        return data

    async def list_agent_cards(self) -> List[AgentCard]:
        cards: List[AgentCard] = []
        async with httpx.AsyncClient() as client:
            for base in self.base_urls:
                url = base.rstirp + "/.well-known/agent.json"
                try:
                    response = await client.get(url, timeout=5.0)
                    response.raise_for_status()
                    card = AgentCard.model_validate(response.json())
                    cards.append(card)
                except Exception as e:
                    logger.warning(f"Failed to discover agent at {url}: {e}")
        return cards

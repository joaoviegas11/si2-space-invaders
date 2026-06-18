__author__ = "Mário Antunes"
__version__ = "1.0.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"

import logging
import random
import asyncio
from typing import Optional, Dict, Any
from agents.base_agent import BaseAgent

class AllWinAgent(BaseAgent):
    async def deliberate(self) -> Optional[Dict[str, Any]]:
        if not self.current_state or self.current_state.get("game_over"):
            return None
        print(self.current_state)

        return {"action": "shoot"}

if __name__ == "__main__":
    agent = AllWinAgent()
    asyncio.run(agent.run())

__author__ = "Mário Antunes"
__version__ = "1.0.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"

import asyncio
import logging
import random
from typing import Optional, Dict, Any
from agents.base_agent import BaseAgent

class DummyAgent(BaseAgent):
    async def deliberate(self) -> Optional[Dict[str, Any]]:
        if not self.current_state or self.current_state.get("game_over"):
            return None
        print(self.current_state)
        valid_actions = self.current_state.get("valid_actions", [])
        if valid_actions:
            # Randomly choose one of the valid actions or choose to do nothing (None)
            action = random.choice(valid_actions + [None])
            if action:
                logging.info(f"Deliberated action: {action}")
            return action

        # Fallback to legacy random choice if valid_actions is not provided by server
        choice = random.choice(["move_west", "move_east", "shoot", "idle"])
        action = None
        if choice == "move_west":
            action = {"action": "move", "direction": "WEST"}
        elif choice == "move_east":
            action = {"action": "move", "direction": "EAST"}
        elif choice == "shoot":
            action = {"action": "shoot"}
        
        if action:
            logging.info(f"Deliberated action (fallback): {action}")
        return action

if __name__ == "__main__":
    agent = DummyAgent()
    asyncio.run(agent.run())

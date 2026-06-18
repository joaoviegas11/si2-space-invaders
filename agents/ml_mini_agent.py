import asyncio
import pickle
import neat
import numpy as np
from typing import Optional, Dict, Any
from agents.base_agent import BaseAgent

def extract_features(state):
    width = int(state.get('width', 11))
    height = float(state.get('height', 11))
    player_x = state.get('player_x', 0)
    
    features = np.zeros(2)
    aliens=state.get('aliens', [])
    diving_aliens = [a for a in aliens if a['is_diving']]
    
    if diving_aliens:
        alien=diving_aliens[0]
    else:
        alien = sorted(
        aliens,
        key=lambda a: abs(player_x - a['x'])
        )[0]
    features[0] = (alien['x'] - player_x) / float(width)  
    features[1] = alien['y']  / float(height)  
        
    return features

class NeatAgent(BaseAgent):
    def __init__(self, config_path, winner_path, server_uri="ws://localhost:8765/ws"):
        super().__init__(server_uri)
        
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
        
        with open(winner_path, "rb") as f:
            genome = pickle.load(f)
            
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)

    async def deliberate(self) -> Optional[Dict[str, Any]]:
        if not self.current_state or self.current_state.get("game_over"):
            return None
            
        inputs = extract_features(self.current_state)
        outputs = self.net.activate(inputs)
        action_idx = np.argmax(outputs)
        
        action = None
        if action_idx == 0:
            action = {"action": "move", "direction": "WEST"}
        elif action_idx == 1:
            action = {"action": "move", "direction": "EAST"}
        elif action_idx == 2:
            action = {"action": "shoot"}
            
        return action

if __name__ == "__main__":
    agent = NeatAgent("config-mini.txt", "winner_mini.pkl")
    asyncio.run(agent.run())
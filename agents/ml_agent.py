import asyncio
import pickle
import neat
import numpy as np
from typing import Optional, Dict, Any
from agents.base_agent import BaseAgent

def extract_features(state):
    width = float(state.get('width', 11.0))
    height = float(state.get('height', 11.0))
    player_x = float(state.get('player_x', 0.0))
    aliens = state.get('aliens', [])
    
    features = np.zeros(4, dtype=np.float32)
    
    if not aliens:
        return features
        
    diving_aliens = [a for a in aliens if a.get('is_diving', False)]
    
    if diving_aliens:
        target = min(diving_aliens, key=lambda a: a['y'])
        features[0] = target['y'] / height
    else:
        features[0] = 1.0 
        target = min(aliens, key=lambda a: abs(player_x - a['x']))
        
    dx = target['x'] - player_x
    features[1] = dx / width
    
    can_shoot = any(action.get("action") == 'shoot' for action in state.get('valid_actions', []))
    features[2] = 1.0 if can_shoot else 0.0
    
    features[3] = 1.0 if abs(dx) < 0.8 else 0.0

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
    agent = NeatAgent("config.txt", "winner.pkl")
    asyncio.run(agent.run())
import asyncio
import pickle
import neat
import numpy as np
from typing import Optional, Dict, Any
from agents.base_agent import BaseAgent

def extract_features(state):
    features = np.zeros(14)
    width = state.get('width', 11)
    height = state.get('height', 11)
    
    features[0] = state.get('player_x', 0) / width
    features[1] = state.get('player_y', 0) / height
    features[2] = state.get('lives', 0) / 5.0
    features[3] = min(state.get('score', 0) / 5000.0, 1.0)
    
    lasers = state.get('lasers', [])
    features[4] = 1.0 if lasers else 0.0
    features[5] = lasers[0]['y'] / height if lasers else 0.5
    
    aliens = state.get('aliens', [])
    if aliens:
        closest_alien = min(aliens, key=lambda a: a['y'])
        features[6] = closest_alien['x'] / width
        features[7] = closest_alien['y'] / height
        features[8] = len(aliens) / 10.0
        features[9] = sum(a['x'] for a in aliens) / (len(aliens) * width)
        features[10] = sum(a['y'] for a in aliens) / (len(aliens) * height)
        features[11] = min(a['y'] for a in aliens) / height
        features[12] = sum(1 for a in aliens if a['x'] < width/2) / 5.0
        features[13] = sum(1 for a in aliens if a['x'] >= width/2) / 5.0
        
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
    agent = NeatAgent("config-feedforward.txt", "winner.pkl")
    asyncio.run(agent.run())
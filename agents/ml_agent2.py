import asyncio
import pickle
import neat
import numpy as np
from typing import Optional, Dict, Any
from agents.base_agent import BaseAgent

def extract_features(state):
    # AGORA SÃO 11 INPUTS!
    features = np.zeros(11) 
    width = state.get('width', 11)
    height = state.get('height', 11)
    
    # 1. Posição do jogador
    features[0] = state.get('player_x', width/2) / width
    
    # 2. Informação sobre Lasers
    lasers = state.get('lasers', [])
    features[1] = 1.0 if lasers else 0.0
    if lasers:
        features[2] = lasers[0]['x'] / width
        features[3] = lasers[0]['y'] / height
        
    # 3. Análise da Ameaça
    aliens = state.get('aliens', [])
    if aliens:
        diving_aliens = [a for a in aliens if a.get('is_diving')]
        static_aliens = [a for a in aliens if not a.get('is_diving')]
        
        if static_aliens:
            closest_static = min(static_aliens, key=lambda a: a['y'])
            features[4] = closest_static['x'] / width
            features[5] = closest_static['y'] / height
            
        if diving_aliens:
            closest_diver = min(diving_aliens, key=lambda a: a['y'])
            features[6] = closest_diver['x'] / width
            features[7] = closest_diver['y'] / height
            features[8] = (closest_diver['x'] - state.get('player_x', 0)) / width
            
        features[9] = len(aliens) / 10.0

    # ----------------------------------------------------
    # 4. NOVO: SENSOR DE COOLDOWN (A arma está pronta?)
    # Verifica se a ação "shoot" está nas valid_actions que o servidor envia
    valid_actions = state.get("valid_actions", [])
    can_shoot = any(act.get("action") == "shoot" for act in valid_actions)
    features[10] = 1.0 if can_shoot else 0.0
    # ----------------------------------------------------
        
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
    agent = NeatAgent("config_finetune.txt", "winner_075_v4.pkl")
    # agent = NeatAgent("config_finetune.txt", "winner_100_v4.pkl")
    # agent = NeatAgent("config.txt", "winner_goated_behaviour.pkl")
    asyncio.run(agent.run())
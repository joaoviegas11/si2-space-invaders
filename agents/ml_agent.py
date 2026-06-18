import asyncio
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
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

class PyTorchAgent(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=6, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.fc2(self.relu(self.fc1(x)))

    def predict_and_observe(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            
        in_layer = x
        hidden_layer = self.relu(self.fc1(in_layer))
        out_layer = self.fc2(hidden_layer)
        
        return out_layer, [in_layer.tolist(), hidden_layer.tolist(), out_layer.tolist()]

    def update(self, weights_array: np.ndarray) -> None:
        idx = 0
        for param in self.parameters():
            param_len = param.data.numel()
            new_weights = weights_array[idx : idx + param_len]
            param.data = torch.tensor(new_weights).view(param.data.size()).float()
            idx += param_len

class PyTorchSpaceAgent(BaseAgent):
    def __init__(self, weights_path, server_uri="ws://localhost:8765/ws"):
        super().__init__(server_uri)
        self.model = PyTorchAgent()
        
        with open(weights_path, "r") as f:
            weights = json.load(f)
            self.model.update(np.array(weights))

    def draw_network_cv2(self, activations, action_idx):
        img = np.zeros((500, 800, 3), dtype=np.uint8)
        
        layers = [
            (activations[0], 150, "Entrada (Sensores)"),
            (activations[1], 400, "Oculta (ReLU)"),
            (activations[2], 650, "Saida (Acoes)")
        ]

        for layer_idx, (layer_vals, x, name) in enumerate(layers):
            num_nodes = len(layer_vals)
            spacing = 400 // max(1, num_nodes)
            start_y = 50 + (400 - spacing * num_nodes) // 2

            cv2.putText(img, name, (x - 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            for i, val in enumerate(layer_vals):
                y = start_y + i * spacing
                
                if layer_idx == 0:
                    intensity = min(255, max(0, int(val * 255)))
                    color = (0, intensity, 0)
                elif layer_idx == 1:
                    color = (255, 150, 0) if val > 0 else (40, 40, 40)
                else:
                    if i == action_idx:
                        color = (0, 0, 255)
                    else:
                        color = (40, 40, 40)

                cv2.circle(img, (x, y), 12, color, -1)
                cv2.circle(img, (x, y), 12, (150, 150, 150), 1)

        actions_str = ["OESTE", "ESTE", "DISPARAR"]
        action_text = f"Decisao: Mover para {actions_str[action_idx]}" if action_idx != 2 else "Decisao: DISPARAR LASER"
        cv2.putText(img, action_text, (250, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Neural Network Dashboard", img)
        cv2.waitKey(1)

    async def deliberate(self) -> Optional[Dict[str, Any]]:
        if not self.current_state or self.current_state.get("game_over"):
            return None
            
        inputs = extract_features(self.current_state)
        
        with torch.no_grad():
            outputs, activations = self.model.predict_and_observe(inputs)
            action_idx = torch.argmax(outputs).item()
        
        self.draw_network_cv2(activations, action_idx)
        
        action = None
        if action_idx == 0:
            action = {"action": "move", "direction": "WEST"}
        elif action_idx == 1:
            action = {"action": "move", "direction": "EAST"}
        elif action_idx == 2:
            action = {"action": "shoot"}
            
        return action

if __name__ == "__main__":
    agent = PyTorchSpaceAgent("best_pytorch_weights.json")
    try:
        asyncio.run(agent.run())
    finally:
        cv2.destroyAllWindows()
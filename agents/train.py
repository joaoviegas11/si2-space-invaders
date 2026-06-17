import json
import numpy as np
import torch
import torch.nn as nn
import cv2
from sko.PSO import PSO
import server.logic as logic

def extract_features(state):
    width = int(state.get('width', 11))
    height = float(state.get('height', 11))
    player_x = state.get('player_x', 0)
    
    features = np.zeros(6)
    features[0] = player_x / float(width)
    
    lasers = state.get('lasers', [])
    features[1] = 1.0 if lasers else 0.0
    features[2] = lasers[0]['y'] / height if lasers else 1.0
    
    alien_columns = np.zeros(width)
    diving_aliens = []
    
    for alien in state.get('aliens', []):
        col = int(round(alien['x']))
        if 0 <= col < width:
            alien_columns[col] = 1
                
        if alien.get('is_diving', False):
            diving_aliens.append(alien)
                
    # features[3:14] = alien_columns[:11]
    
    if diving_aliens:
        closest_diver = min(diving_aliens, key=lambda a: a['y'])
        
        features[3] = 1.0 
        
        features[4] = (closest_diver['x'] - player_x) / float(width)
        
        features[5] = closest_diver['y'] / height
    else:
        features[3] = 0.0
        features[4] = 0.0
        features[5] = 0.0
        
    return features

class PyTorchAgent(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=5, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.net(x)

    def ravel(self) -> np.ndarray:
        weights = []
        for param in self.parameters():
            weights.extend(param.data.cpu().numpy().flatten())
        return np.array(weights)

    def update(self, weights_array: np.ndarray) -> None:
        idx = 0
        for param in self.parameters():
            param_len = param.data.numel()
            new_weights = weights_array[idx : idx + param_len]
            param.data = torch.tensor(new_weights).view(param.data.size()).float()
            idx += param_len

agent_model = PyTorchAgent()

global_best_score = 0 

def evaluate_model(params):
    global global_best_score
    
    agent_model.update(params)
    env = logic.SpaceInvaders()
    fixed_dt = 1.0 / 30.0
    max_steps = 3000
    steps = 0
    score = 0
    
    penalidade = 0.0
    vidas_anteriores = env.lives

    while not env.game_over and steps < max_steps:
        state = env.get_state()
        inputs = extract_features(state)
        
        with torch.no_grad():
            outputs = agent_model(inputs)
            action_idx = torch.argmax(outputs).item()
        
        # # --- LÓGICA DE PENALIDADE DE FUGA ---
        # player_col = int(round(state.get('player_x', 0)))
        # width = int(state.get('width', 11))
        
        # if 0 <= player_col < width:
        #     distancia_perigo = inputs[3 + player_col]
        #     if distancia_perigo < 0.4 and action_idx == 2:
        #         penalidade += 2.0 
        
        if action_idx == 0:
            env.move_player("WEST")
        elif action_idx == 1:
            env.move_player("EAST")
        elif action_idx == 2:
            env.shoot_laser()

        env.update(fixed_dt)
        if env.score < score:
            penalidade += 10.0
        score=env.score 
        steps += 1
        
        # --- PENALIDADE POR PERDA DE VIDA ---
        if env.lives < vidas_anteriores:
            penalidade += 100.0
            vidas_anteriores = env.lives

    # Atualiza a pontuação máxima global se esta nave bateu o recorde
    if env.score > global_best_score:
        global_best_score = env.score
            
    fitness = env.score+( round(steps/30,2))/10 - penalidade
    return -fitness

def draw_graph_cv2(historico, max_iter, current_iter, best_score):
    """Função auxiliar para desenhar o gráfico com OpenCV"""
    # Criar ecrã escuro: 500px altura x 800px largura
    img = np.ones((500, 800, 3), dtype=np.uint8) * 30
    
    # Margens do gráfico
    pad_l, pad_r, pad_t, pad_b = 80, 40, 80, 70
    graph_w = 800 - pad_l - pad_r
    graph_h = 500 - pad_t - pad_b
    
    # Título e eixos
    cv2.putText(img, "Curva de Aprendizagem (OpenCV)", (pad_l, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(img, (pad_l, 500 - pad_b), (800 - pad_r, 500 - pad_b), (150, 150, 150), 2) # Eixo X
    cv2.line(img, (pad_l, pad_t), (pad_l, 500 - pad_b), (150, 150, 150), 2) # Eixo Y
    
    if len(historico) > 0:
        max_val = max(historico)
        min_val = min(historico)
        
        # Prevenir divisão por zero se todos os valores forem iguais
        if max_val == min_val:
            max_val += 1.0
            min_val -= 1.0
            
        points = []
        for j, val in enumerate(historico):
            # Mapear a geração (j) para o pixel X
            x = int(pad_l + (j / max(1, max_iter - 1)) * graph_w)
            # Mapear o valor (val) para o pixel Y (invertido, porque 0 é no topo do ecrã)
            y = int((500 - pad_b) - ((val - min_val) / (max_val - min_val)) * graph_h)
            points.append((x, y))
            
        # Desenhar as linhas a unir os pontos
        for j in range(1, len(points)):
            cv2.line(img, points[j-1], points[j], (0, 255, 255), 2) # Linha amarela
            cv2.circle(img, points[j], 4, (0, 150, 255), -1) # Ponto laranja
            
        # Mostrar o valor atual no fundo do ecrã
        info_text = f"Geracao: {current_iter}/{max_iter} | Melhor Fitness (Matematica): {historico[-1]:.2f}"
        score_text = f"Melhor Pontuacao Pura (Jogo): {best_score}"
        
        cv2.putText(img, info_text, (pad_l, 500 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, score_text, (pad_l, 500 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) # Amarelo para destacar
        
    cv2.imshow("Treino PSO", img)
    cv2.waitKey(1) # Força o OpenCV a desenhar o frame imediatamente

def run_evolution():
    num_params = len(agent_model.ravel())
    lb = [-5.0] * num_params
    ub = [5.0] * num_params
    
    print(f"A iniciar otimizacao PSO via sko para {num_params} parametros...")
    
    pso = PSO(
        func=evaluate_model, 
        n_dim=num_params, 
        pop=50, 
        max_iter=15, 
        lb=lb, 
        ub=ub, 
        w=0.8, 
        c1=0.5, 
        c2=0.5
    )
    
    historico_melhores = []
    
    # Executar o treino geração a geração
    for i in range(pso.max_iter):
        pso.update_V()
        pso.recorder()
        pso.update_X()
        pso.cal_y()
        pso.update_pbest()
        pso.update_gbest()
        pso.gbest_y_hist.append(pso.gbest_y)
        
        melhor_fitness_geracao = -pso.gbest_y[0] if isinstance(pso.gbest_y, (list, np.ndarray)) else -pso.gbest_y
        historico_melhores.append(melhor_fitness_geracao)
        
        print(f"Geração {i+1}/{pso.max_iter} concluída -> Melhor Fitness: {melhor_fitness_geracao:.2f} | Pontuação Máx: {global_best_score}")
        
        # Desenhar o gráfico usando OpenCV de forma leve (agora a passar a pontuação pura também)
        draw_graph_cv2(historico_melhores, pso.max_iter, i+1, global_best_score)
    
    with open("best_pytorch_weights.json", "w") as f:
        json.dump(pso.gbest_x.tolist(), f)
        
    melhor_final = -pso.gbest_y[0] if isinstance(pso.gbest_y, (list, np.ndarray)) else -pso.gbest_y
    print(f"\nTreino terminado! Pesos guardados. Melhor pontuação absoluta: {melhor_final:.2f}")
    
    print("Pressione qualquer tecla na janela do gráfico para sair.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    run_evolution()
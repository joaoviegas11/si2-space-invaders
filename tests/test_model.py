import neat
import pickle
import numpy as np
import server.logic as logic

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

def avaliar_modelo(nome, config_path, winner_path, feature_extractor, num_jogos=10, max_steps=5000):
    print(f"\n--- A avaliar Modelo: {nome} ---")
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    with open(winner_path, "rb") as f:
        genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    resultados_score = []
    resultados_steps = []
    resultados_vidas = []

    fixed_dt = 1.0 / 30.0

    for i in range(num_jogos):
        env = logic.SpaceInvaders()
        steps = 0
        
        while not env.game_over and steps < max_steps:
            state = env.get_state()
            inputs = feature_extractor(state)
            
            outputs = net.activate(inputs)
            action_idx = np.argmax(outputs)
            
            if action_idx == 0:
                env.move_player("WEST")
            elif action_idx == 1:
                env.move_player("EAST")
            elif action_idx == 2:
                env.shoot_laser()
                
            env.update(fixed_dt)
            steps += 1
            
        resultados_score.append(env.score)
        resultados_steps.append(steps)
        resultados_vidas.append(env.lives)
        
        print(f"Jogo {i+1}/{num_jogos} -> Score: {env.score:5} | Vidas Restantes: {env.lives} | Passos (Tempo): {steps} steps")

    return {
        "nome": nome,
        "score_medio": np.mean(resultados_score),
        "vidas_medias": np.mean(resultados_vidas),
        "tempo_medio": np.mean(resultados_steps)
    }

if __name__ == "__main__":
    limite_de_tempo = 100000
    
    stats = avaliar_modelo(
        nome="NEAT(4 Entradas)",
        config_path="config.txt", 
        winner_path="winner.pkl", 
        feature_extractor=extract_features,
        num_jogos=10,
        max_steps=limite_de_tempo
    )
    

    print("\n" + "="*50)
    print("           RESULTADOS FINAIS (MÉDIA DE 10 JOGOS)           ")
    print("="*50)
    print(f"{'Modelo':<28} | {'Score':<8} | {'Vidas':<5} | {'Tempo'}")
    print("-" * 50)
    print(f"{stats['nome']:<28} | {stats['score_medio']:<8.1f} | {stats['vidas_medias']:<5.1f} | {stats['tempo_medio']:.0f}")
    print("="*50)
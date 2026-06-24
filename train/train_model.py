import neat
import server.logic as logic
import numpy as np
import pickle

import numpy as np

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

def eval_single_game(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = logic.SpaceInvaders()
    
    fixed_dt = 1.0 / 30.0
    max_steps = 3000
    steps = 0
    fitness = 0.0
    
    x_heatmap = np.zeros(env.width)
    
    while not env.game_over and steps < max_steps:
        state = env.get_state()
        inputs = extract_features(state)

        current_x_int = int(min(max(env.player_x, 0), env.width - 1))
        x_heatmap[current_x_int] += 1

        aliens = state.get('aliens', [])
        if aliens:
            diving_aliens = [a for a in aliens if a.get('is_diving', False)]
            if diving_aliens:
                closest_diver = diving_aliens[0]
                if closest_diver['y'] < 4.0:
                    dist_x = abs(env.player_x - closest_diver['x'])
                    evasao = min(1.0, dist_x / 3.0) 
                    fitness += evasao * 0.2
            else:
                closest_alien = sorted(aliens, key=lambda a: abs(env.player_x - a['x']))[0]
                dist_x = abs(env.player_x - closest_alien['x'])
                alinhamento = max(0.0, 1.0 - (dist_x / env.width))
                fitness += alinhamento * 0.2
        
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
        
    fitness += (env.score * 10.0)
    
    fitness -= (3 - env.lives) * 200
    if env.lives <= 0:
        fitness -= 2000

    if steps > 0:
        max_time_in_one_spot = np.max(x_heatmap) / steps
        
        if max_time_in_one_spot > 0.45:
            fitness *= 0.1

    return fitness if fitness > 0 else 0.0

def eval_genomes(genomes, config):
    episodes_per_genome = 3 
    
    for genome_id, genome in genomes:
        fitness_scores = []
        for _ in range(episodes_per_genome):
            score = eval_single_game(genome, config)
            fitness_scores.append(score)
            
        genome.fitness = sum(fitness_scores) / episodes_per_genome

def run_evolution(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    p = neat.Population(config)
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    print("A iniciar o treino NEAT com 4 neurónios de entrada")
    winner = p.run(eval_genomes, 10)
    
    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
        
    print("\nTreino concluído. Guardado em 'winner.pkl'")
    return winner

if __name__ == "__main__":
    run_evolution("config.txt")
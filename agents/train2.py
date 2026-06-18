import neat
import server.logic as logic
import numpy as np
import pickle

def extract_features(state):
    features = np.zeros(11) 
    width = state.get('width', 11)
    height = state.get('height', 11)
    
    features[0] = state.get('player_x', width/2) / width
    
    lasers = state.get('lasers', [])
    features[1] = 1.0 if lasers else 0.0
    if lasers:
        features[2] = lasers[0]['x'] / width
        features[3] = lasers[0]['y'] / height
        
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

    valid_actions = state.get("valid_actions", [])
    can_shoot = any(act.get("action") == "shoot" for act in valid_actions)
    features[10] = 1.0 if can_shoot else 0.0
        
    return features

def eval_single_game(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = logic.SpaceInvaders()
    
    fixed_dt = 1.0 / 30.0
    max_steps = 6000
    steps = 0
    fitness = 0.0
    
    x_heatmap = np.zeros(env.width)
    
    while not env.game_over and steps < max_steps:
        state = env.get_state()
        inputs = extract_features(state)

        current_x_int = int(min(env.player_x, env.width - 1))
        x_heatmap[current_x_int] += 1

        aliens = state.get('aliens', [])
        if aliens:
            diving_aliens = [a for a in aliens if a.get('is_diving')]
            if diving_aliens:
                closest_diver = min(diving_aliens, key=lambda a: a['y'])
                if closest_diver['y'] < 4.0:
                    dist_x = abs(state.get('player_x', 0) - closest_diver['x'])
                    evasao = min(1.0, dist_x / 3.0) 
                    fitness += evasao * 0.2
                else:
                    dist_x = abs(state.get('player_x', 0) - closest_diver['x'])
                    alinhamento = max(0.0, 1.0 - (dist_x / env.width))
                    fitness += alinhamento * 0.2
            else:
                closest_alien = min(aliens, key=lambda a: a['y'])
                dist_x = abs(state.get('player_x', 0) - closest_alien['x'])
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
        
    fitness += (env.score * 5.0) - (steps * 0.5)
    fitness -= (3 - env.lives) * 500
    if env.lives <= 0:
        fitness -= 1000

    if steps > 0:
        max_time_in_one_spot = np.max(x_heatmap) / steps
        
        if max_time_in_one_spot > 0.45:
            fitness *= 0.1

    fitness = fitness if fitness > 0 else 0
        
    return fitness

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
    
    winner = p.run(eval_genomes, 100)
    
    with open("winner_2.pkl", "wb") as f:
        pickle.dump(winner, f)
        
    print("\nTreino concluído. Campeão guardado em 'winner.pkl'")
    return winner

if __name__ == "__main__":
    run_evolution("config.txt")
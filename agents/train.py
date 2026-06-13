import neat
import server.logic as logic
import numpy as np
import pickle

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

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        env = logic.SpaceInvaders()
        fixed_dt = 1.0 / 30.0
        max_steps = 3000
        steps = 0
        
        while not env.game_over and steps < max_steps:
            state = env.get_state()
            inputs = extract_features(state)
            
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
            
        genome.fitness = env.score + 0.1 * steps

def run_evolution(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    p = neat.Population(config)
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    winner = p.run(eval_genomes, 50)
    
    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
        
    return winner

if __name__ == "__main__":
    run_evolution("config-feedforward.txt")
__author__ = "Mário Antunes"
__version__ = "1.0.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"

import math
import random
from typing import Dict, List, Any

class Laser:
    def __init__(self, x: float, y: float, speed: float = 8.0):
        self.x = x
        self.y = y
        self.speed = speed
        self.active = True

    def update(self, dt: float):
        self.y += self.speed * dt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "speed": self.speed,
            "active": self.active
        }

class Alien:
    def __init__(self, index: int, base_x: float, base_y: float):
        self.index = index
        self.base_x = base_x
        self.base_y = base_y
        
        # Desynchronized horizontal figure-8 amplitudes, frequency and phase
        self.amp_x = 0.8 + 0.1 * (index % 3)
        self.amp_y = 0.5 + 0.05 * (index % 2)
        self.freq = 1.0 + 0.15 * (index % 4)
        self.phase = (2.0 * math.pi / 10.0) * index
        
        # Dynamic float positions
        self.x = base_x
        self.y = base_y
        
        # Diving state variables
        self.is_diving = False
        self.target_x = 0.0
        self.dive_speed = 5.0
        
        self.active = True

    def update(self, dt: float, time_elapsed: float, base_shift_x: float = 0.0, player_x: float | None = None, difficulty_scale: float = 1.0):
        if not self.active:
            return

        if not self.is_diving:
            # Figure-8 parametric motion + global shift
            self.x = (self.base_x + base_shift_x) + self.amp_x * math.sin(self.freq * difficulty_scale * time_elapsed + self.phase)
            self.y = self.base_y + self.amp_y * math.sin(2.0 * self.freq * difficulty_scale * time_elapsed + self.phase)
        else:
            # Homing tracking: if the alien is above y = 3.0, track the player's current x position
            if player_x is not None and self.y > 3.0:
                self.target_x = player_x

            # Constant downward speed
            vy = -self.dive_speed
            
            # Predictive Interception: Calculate horizontal velocity needed to hit target at y=0
            time_to_ground = max(0.1, self.y / self.dive_speed)
            vx = (self.target_x - self.x) / time_to_ground
            
            # Cap horizontal speed to 1.5x dive speed to prevent teleporting
            max_vx = self.dive_speed * 1.5
            vx = max(-max_vx, min(max_vx, vx))
            
            self.x += vx * dt
            self.y += vy * dt

    def start_dive(self, player_x: float, speed: float = 8.0):
        self.is_diving = True
        self.target_x = player_x
        self.dive_speed = speed

    def reset_dive(self):
        self.is_diving = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "x": self.x,
            "y": self.y,
            "is_diving": self.is_diving,
            "active": self.active
        }

class SpaceInvaders:
    def __init__(self, width: int = 11, height: int = 11, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.high_score = 0
        self.reset_game()

    def reset_game(self):
        self.player_x: float = float(self.width // 2)
        self.player_y: float = 0.0
        self.lives = 3
        self.score = 0
        self.checkpoint_score = 0
        self.game_over = False
        
        self.aliens: List[Alien] = []
        self.lasers: List[Laser] = []
        
        self.time_elapsed = 0.0
        self.dive_cooldown = 3.0
        self.time_since_last_dive = 0.0
        
        self.shoot_cooldown = 0.5
        self.time_since_last_shoot = self.shoot_cooldown
        
        self._init_aliens()

    def _init_aliens(self):
        self.aliens = []
        
        # 2 rows of 5 aliens
        # Columns centered nicely: 2.0, 3.5, 5.0, 6.5, 8.0
        columns = [2.0, 3.5, 5.0, 6.5, 8.0]
        rows = [8.0, 9.0]
        
        index = 0
        for y_base in rows:
            for x_base in columns:
                self.aliens.append(Alien(index, x_base, y_base))
                index += 1

    def move_player(self, direction: str):
        if self.game_over:
            return

        if direction == "WEST":
            if self.player_x > 0.0:
                self.player_x -= 1.0
        elif direction == "EAST":
            if self.player_x < self.width - 1.0:
                self.player_x += 1.0

    def shoot_laser(self):
        if self.game_over:
            return

        if self.time_since_last_shoot >= self.shoot_cooldown:
            # Firing laser from the player's front
            self.lasers.append(Laser(self.player_x, 1.0, speed=8.0))
            self.time_since_last_shoot = 0.0

    def _die(self):
        self.lives -= 1
        self.score = self.checkpoint_score
        self.player_x = float(self.width // 2)
        self.lasers = []
        
        # Reset any diving alien
        for alien in self.aliens:
            alien.reset_dive()
            
        if self.lives <= 0:
            self.game_over = True

    def update(self, dt: float):
        if self.game_over:
            return

        self.time_elapsed += dt
        self.time_since_last_shoot += dt
        self.time_since_last_dive += dt

        # 1. Update Lasers
        active_lasers = []
        for laser in self.lasers:
            laser.update(dt)
            if laser.y < self.height:
                active_lasers.append(laser)
        self.lasers = active_lasers

        # 2. Update Aliens
        # Calculate difficulty scale based on current dive speed (8.0 base -> 12.0 max)
        current_dive_speed = min(12.0, 8.0 + 0.01 * self.score)
        difficulty_scale = current_dive_speed / 8.0
        
        sweep_offset = 1.5 * math.sin(1.2 * difficulty_scale * self.time_elapsed)
        for alien in self.aliens:
            alien.update(dt, self.time_elapsed, base_shift_x=sweep_offset, player_x=self.player_x, difficulty_scale=difficulty_scale)

        # 3. Schedule Alien Dives
        current_cooldown = max(1.5, self.dive_cooldown - 0.005 * self.score)
        # print(self.score,current_dive_speed,difficulty_scale,current_cooldown)

        if self.time_since_last_dive >= current_cooldown:
            available_aliens = [a for a in self.aliens if a.active and not a.is_diving]
            if available_aliens:
                diving_alien = random.choice(available_aliens)
                diving_alien.start_dive(self.player_x, speed=current_dive_speed)
            self.time_since_last_dive = 0.0

        # 4. Check Laser-Alien Collisions (Distance < 0.8) using numpy
        if self.lasers and self.aliens:
            import numpy as np
            active_aliens = [a for a in self.aliens if a.active]
            if active_aliens:
                laser_coords = np.array([[las.x, las.y] for las in self.lasers])
                alien_coords = np.array([[a.x, a.y] for a in active_aliens])
                
                diff = laser_coords[:, np.newaxis, :] - alien_coords[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=-1))
                
                collisions = dists < 0.8
                laser_hit_idx, alien_hit_idx = np.where(collisions)
                
                hit_lasers = set()
                hit_aliens = set()
                for l_idx, a_idx in zip(laser_hit_idx, alien_hit_idx):
                    if l_idx not in hit_lasers and a_idx not in hit_aliens:
                        active_aliens[a_idx].active = False
                        active_aliens[a_idx].is_diving = False
                        self.lasers[l_idx].active = False
                        hit_lasers.add(l_idx)
                        hit_aliens.add(a_idx)
                        self.score += 10
                        if self.score > self.high_score:
                            self.high_score = self.score
                self.lasers = [las for las in self.lasers if las.active]

        # 5. Check Alien-Player Collisions (Distance < 0.8) and Alien-Bottom Misses using numpy
        diving_aliens = [a for a in self.aliens if a.active and a.is_diving]
        if diving_aliens:
            import numpy as np
            alien_coords = np.array([[a.x, a.y] for a in diving_aliens])
            player_coord = np.array([self.player_x, self.player_y])
            
            dists = np.sqrt(np.sum((alien_coords - player_coord)**2, axis=-1))
            if np.any(dists < 0.8):
                self._die()
                return
                
            for alien in diving_aliens:
                if alien.y <= 0.0:
                    alien.reset_dive()
                    self.score = max(self.checkpoint_score, self.score - 10)

        # 6. Check Wave Cleared
        active_count = len([a for a in self.aliens if a.active])
        if active_count == 0:
            self.score += 100
            if self.score > self.high_score:
                self.high_score = self.score
            self.checkpoint_score = self.score
            # Respawn wave
            self._init_aliens()

    def get_state(self) -> Dict[str, Any]:
        valid_actions = []
        if not self.game_over:
            if self.player_x > 0.0:
                valid_actions.append({"action": "move", "direction": "WEST"})
            if self.player_x < self.width - 1.0:
                valid_actions.append({"action": "move", "direction": "EAST"})
            if self.time_since_last_shoot >= self.shoot_cooldown:
                valid_actions.append({"action": "shoot"})

        return {
            "width": self.width,
            "height": self.height,
            "player_x": self.player_x,
            "player_y": self.player_y,
            "lives": self.lives,
            "score": self.score,
            "high_score": self.high_score,
            "game_over": self.game_over,
            "lasers": [las.to_dict() for las in self.lasers],
            "aliens": [a.to_dict() for a in self.aliens if a.active],
            "valid_actions": valid_actions
        }

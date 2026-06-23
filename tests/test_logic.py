__author__ = "Mário Antunes"
__version__ = "1.0.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"

import unittest
from server.logic import SpaceInvaders, Laser


class TestSpaceInvadersLogic(unittest.TestCase):
    game: SpaceInvaders = SpaceInvaders()

    def setUp(self):
        self.game = SpaceInvaders(width=11, height=11)

    def test_initial_state(self):
        self.assertEqual(self.game.player_x, 5.0)
        self.assertEqual(self.game.player_y, 0.0)
        self.assertEqual(self.game.lives, 3)
        self.assertEqual(self.game.score, 0)
        self.assertEqual(self.game.checkpoint_score, 0)
        self.assertFalse(self.game.game_over)
        self.assertEqual(len(self.game.aliens), 10)

    def test_move_bounds(self):
        # Move WEST repeatedly
        for _ in range(10):
            self.game.move_player("WEST")
        self.assertEqual(self.game.player_x, 0.0)

        # Move EAST repeatedly
        for _ in range(20):
            self.game.move_player("EAST")
        self.assertEqual(self.game.player_x, 10.0)

    def test_shoot_cooldown(self):
        self.game.shoot_laser()
        self.assertEqual(len(self.game.lasers), 1)

        # Try shooting immediately again (should fail due to cooldown)
        self.game.shoot_laser()
        self.assertEqual(len(self.game.lasers), 1)

        # Fast forward time past cooldown
        self.game.update(0.6)
        self.game.shoot_laser()
        self.assertEqual(len(self.game.lasers), 2)

    def test_laser_alien_collision(self):
        self.game.lasers = [Laser(x=5.0, y=7.5)]

        # Deactivate all aliens except index 0 and 2
        # Index 0 is kept active and static to prevent wave clearing (active_count drops to 0)
        # while being far enough to not interfere with the collision.
        for i, alien in enumerate(self.game.aliens):
            if i not in (0, 2):
                alien.active = False
            else:
                alien.active = True
                alien.amp_x = 0.0
                alien.amp_y = 0.0

        # Position alien 2 exactly at (5.0, 8.3)
        self.game.aliens[2].base_x = 5.0
        self.game.aliens[2].base_y = 8.3

        self.game.update(0.1)  # Updates positions, laser flies up to (5.0, 8.3)

        self.assertFalse(self.game.aliens[2].active)
        self.assertEqual(self.game.score, 10)
        self.assertEqual(len(self.game.lasers), 0)

    def test_diving_hit_death(self):
        # Advance score and checkpoint
        self.game.score = 50
        self.game.checkpoint_score = 50

        # Trigger an alien dive and position it close to player center (5.0, 0.0)
        alien = self.game.aliens[0]
        alien.active = True
        alien.start_dive(self.game.player_x, speed=5.0)
        alien.x = 5.0
        alien.y = 0.5

        # Trigger update (should register collision and death)
        self.game.update(0.1)

        self.assertEqual(self.game.lives, 2)
        self.assertEqual(self.game.score, 50)  # Reset to checkpoint score
        self.assertEqual(self.game.player_x, 5.0)  # Reset player center
        self.assertFalse(alien.is_diving)

    def test_diving_miss_penalty(self):
        # Increase score, checkpoint stays 0
        self.game.score = 30
        self.game.checkpoint_score = 0

        alien = self.game.aliens[0]
        alien.active = True
        alien.start_dive(self.game.player_x, speed=5.0)
        alien.x = 2.0
        alien.y = 0.05

        # Update alien y position so it drops <= 0.0
        alien.update(
            0.05, 0.0
        )  # Will move y down since dive speed is 5.0 and dy is negative

        self.game.update(0.01)  # Triggers miss penalty check

        self.assertEqual(self.game.score, 20)  # Deduced by 10 points
        self.assertFalse(alien.is_diving)  # Teleports/resets dive

    def test_wave_cleared(self):
        # Setup all aliens as inactive
        for alien in self.game.aliens:
            alien.active = False

        # Update frame (triggers wave cleared checks)
        self.game.update(0.1)

        self.assertEqual(self.game.score, 100)  # Gets +100 bonus
        self.assertEqual(self.game.checkpoint_score, 100)  # Checkpoint updated

        # Squad respawned
        self.assertEqual(len([a for a in self.game.aliens if a.active]), 10)

    def test_valid_actions(self):
        # Initial state (player_x = 5.0, time_since_last_shoot >= shoot_cooldown)
        state = self.game.get_state()
        self.assertIn("valid_actions", state)
        self.assertEqual(len(state["valid_actions"]), 3)
        self.assertIn({"action": "move", "direction": "WEST"}, state["valid_actions"])
        self.assertIn({"action": "move", "direction": "EAST"}, state["valid_actions"])
        self.assertIn({"action": "shoot"}, state["valid_actions"])

        # Move WEST repeatedly to bounds
        for _ in range(10):
            self.game.move_player("WEST")
        state = self.game.get_state()
        # Should not contain WEST
        self.assertNotIn(
            {"action": "move", "direction": "WEST"}, state["valid_actions"]
        )
        self.assertIn({"action": "move", "direction": "EAST"}, state["valid_actions"])

        # Move EAST repeatedly to bounds
        for _ in range(20):
            self.game.move_player("EAST")
        state = self.game.get_state()
        # Should not contain EAST
        self.assertNotIn(
            {"action": "move", "direction": "EAST"}, state["valid_actions"]
        )
        self.assertIn({"action": "move", "direction": "WEST"}, state["valid_actions"])

        # Shoot to trigger cooldown
        self.game.shoot_laser()
        state = self.game.get_state()
        # Should not contain shoot
        self.assertNotIn({"action": "shoot"}, state["valid_actions"])

        # Game over state
        self.game.game_over = True
        state = self.game.get_state()
        self.assertEqual(state["valid_actions"], [])

    def test_difficulty_scaling(self):
        # Verify the calculation is working by setting score
        self.game.score = 200
        # When update runs, it computes difficulty_scale and sweep_offset
        self.game.update(0.1)
        
        # Let's test that the alien update was called with difficulty_scale=1.25
        # Since update is called on aliens, they move according to the difficulty_scale.
        # Let's check difficulty_scale directly in update loop or state if needed.
        # But we can verify it by checking that an alien's update is affected by difficulty_scale.
        alien = self.game.aliens[0]
        # Reset alien position, active, and not diving
        alien.active = True
        alien.is_diving = False
        alien.base_x = 5.0
        alien.base_y = 5.0
        # For difficulty_scale = 1.0:
        alien.update(dt=0.1, time_elapsed=1.0, base_shift_x=0.0, player_x=5.0, difficulty_scale=1.0)
        x_1 = alien.x
        
        # Reset and run with difficulty_scale = 2.0:
        alien.x = 5.0
        alien.y = 5.0
        alien.update(dt=0.1, time_elapsed=1.0, base_shift_x=0.0, player_x=5.0, difficulty_scale=2.0)
        x_2 = alien.x
        
        # The movements should be different because of frequency scaling
        self.assertNotEqual(x_1, x_2)

    def test_predictive_interception(self):
        alien = self.game.aliens[0]
        alien.active = True
        # Set a target x far from alien's x
        alien.x = 2.0
        alien.y = 10.0
        alien.start_dive(player_x=8.0, speed=10.0)
        
        # Under predictive interception:
        # time_to_ground = max(0.1, y / dive_speed) = max(0.1, 10.0 / 10.0) = 1.0
        # vx = (target_x - x) / time_to_ground = (8.0 - 2.0) / 1.0 = 6.0
        # Cap is 1.5 * dive_speed = 15.0, so 6.0 is not capped.
        # self.x += vx * dt -> self.x += 6.0 * 0.1 = 0.6 => self.x becomes 2.6
        # self.y += vy * dt -> self.y += -10.0 * 0.1 = -1.0 => self.y becomes 9.0
        
        alien.update(dt=0.1, time_elapsed=1.0, base_shift_x=0.0, player_x=8.0, difficulty_scale=1.0)
        
        self.assertAlmostEqual(alien.x, 2.6)
        self.assertAlmostEqual(alien.y, 9.0)
        
        # Test the speed capping logic:
        # Let's place target_x very far to force vx to exceed max_vx (1.5 * 10.0 = 15.0)
        # e.g., target_x = 100.0. (100.0 - 2.6) / (9.0 / 10.0) = 97.4 / 0.9 = 108.2
        # Capped vx should be 15.0
        # So alien.x should increase by 15.0 * 0.1 = 1.5 => 2.6 + 1.5 = 4.1
        alien.update(dt=0.1, time_elapsed=1.1, base_shift_x=0.0, player_x=100.0, difficulty_scale=1.0)
        self.assertAlmostEqual(alien.x, 4.1)

if __name__ == "__main__":
    unittest.main()


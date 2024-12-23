import gymnasium as gym
import numpy as np

import game_model

class CroissantGameEnv(gym.Env):
    def __init__(self):
        self.game = None
        self.error_count = 0

        self.action_space = gym.spaces.Discrete(2 + len(game_model.config["consume_costs"]))
        self.observation_space = gym.spaces.Box(0, np.inf, shape = (4,), dtype = int)

    def _get_observation(self):
        assert(self.game != None)
        return np.array([self.game.money, self.game.croissants, len(self.game.investments), self.game.turns_left])

    def _get_info(self):
        return { "error_count": self.error_count }

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        self.game = game_model.CroissantGame()
        self.error_count = 0
        return self._get_observation(), self._get_info()

    def step(self, action):
        assert(self.action_space.contains(action))
        try:
            if action == 0:
                self.game.execute_labor()
            elif action == 1:
                self.game.execute_invest()
            else:
                self.game.execute_consume(game_model.config["consume_costs"][action - 2])
        except game_model.InvalidActionException:
            self.error_count += 1

        observation = self._get_observation()
        reward = self.game.croissants - self.error_count

        return observation, reward, self.game.turns_left <= 0, False, self._get_info()

gym.register(id = "bunny-baxter/CroissantGame-v0", entry_point = CroissantGameEnv)

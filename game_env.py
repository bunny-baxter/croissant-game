import gymnasium as gym
import numpy as np

import game_model

class CroissantGameEnv(gym.Env):
    def __init__(self, enable_stash = False):
        self.game = None
        self.error_count = 0

        if enable_stash:
            self.stash_value = 0
        else:
            self.stash_value = None

        action_space_size = 2 + len(game_model.config["consume_costs"])
        if enable_stash:
            action_space_size += 2
        self.action_space = gym.spaces.Discrete(action_space_size)

        observation_space_size = 5
        if enable_stash:
            observation_space_size += 1
        self.observation_space = gym.spaces.Box(0, np.inf, shape = (observation_space_size,), dtype = int)

    def _get_observation(self):
        assert(self.game != None)
        if self.stash_value == None:
            return np.array([self.game.money, self.game.croissants, len(self.game.investments), self.game.turns_left, self.error_count])
        else:
            return np.array([self.game.money, self.game.croissants, len(self.game.investments), self.game.turns_left, self.error_count, self.game.stash_value])

    def _get_info(self):
        return {}

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        if self.game != None:
            self.stash_value = self.game.stash_value
        self.game = game_model.CroissantGame(stash_value = self.stash_value)
        self.error_count = 0
        return self._get_observation(), self._get_info()

    def step(self, action):
        assert(self.action_space.contains(action))
        try:
            if action == 0:
                self.game.execute_labor()
            elif action == 1:
                self.game.execute_invest()
            elif action == 5:
                self.game.execute_stash()
            elif action == 6:
                self.game.execute_unstash()
            else:
                self.game.execute_consume(game_model.config["consume_costs"][action - 2])
        except game_model.InvalidActionException:
            self.error_count += 1
            self.game.execute_noop()

        observation = self._get_observation()
        reward = self.game.croissants - self.error_count + 0.1 * self.game.money

        return observation, reward, self.game.turns_left <= 0, False, self._get_info()

gym.register(id = "bunny-baxter/CroissantGame-v0", entry_point = CroissantGameEnv)
gym.register(id = "bunny-baxter/CroissantGameExploitable-v0", entry_point = lambda: CroissantGameEnv(enable_stash = True))

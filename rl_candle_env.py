import numpy as np
from gymnasium import Env, spaces

class CandleTradeEnv(Env):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.current_step = 10
        self.max_step = len(df) - 2
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)

    def _get_obs(self):
        obs = self.df.iloc[self.current_step - 10:self.current_step][['close', 'rsi14', 'return']].values.flatten()
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        self.current_step = 10
        return self._get_obs(), {}

    def step(self, action):
        reward = 0
        future_return = self.df.iloc[self.current_step + 1]['return']
        if action == 1 and future_return > 0:
            reward = 1
        elif action == 2 and future_return < 0:
            reward = 1
        elif action in [1, 2]:
            reward = -1
        self.current_step += 1
        done = self.current_step >= self.max_step
        return self._get_obs(), reward, done, False, {}

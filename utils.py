from gymnasium import Wrapper

class FrameSkip(Wrapper):
    """A wrapper that skips every 'skip' frames."""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            total_reward += reward
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info
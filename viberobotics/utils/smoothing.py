from abc import ABC, abstractmethod
import numpy as np
from typing import Union

class SmoothingFilter(ABC):
    def __init__(self, default_value: float = 0.0):
        self.default_value = default_value.copy() if isinstance(default_value, np.ndarray) else default_value
        self.value = default_value.copy() if isinstance(default_value, np.ndarray) else default_value
    
    @abstractmethod
    def apply(self, new_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        pass
    
    def reset(self):
        self.value = self.default_value
    
    def get(self) -> Union[float, np.ndarray]:
        return self.value.copy() if isinstance(self.value, np.ndarray) else self.value

class RateLimitedSmoothing(SmoothingFilter):
    def __init__(self, rate_limit: float, default_value: Union[float, np.ndarray] = 0.0):
        super().__init__(default_value)
        self.rate_limit = rate_limit

    def apply(self, new_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        self.value += np.clip(new_value - self.value, -self.rate_limit, self.rate_limit)
        return self.value

class EMASmoothing(SmoothingFilter):
    def __init__(self, alpha: float, default_value: float = 0.0):
        super().__init__(default_value)
        self.alpha = alpha

    def apply(self, new_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
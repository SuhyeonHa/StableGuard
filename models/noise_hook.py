import torch
import torch.nn as nn
import random

class NoiseHook:
    def __init__(self, noise_range=(0.0, 0.5)):
        self.noise_range = noise_range
        self.enabled = False
        self.strength = 0.0

    def __call__(self, module, input, output):
        if not self.enabled:
            return output
        
        self.strength = random.uniform(self.noise_range[0], self.noise_range[1])
        noise = torch.randn_like(output) * self.strength
        
        return output + noise

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
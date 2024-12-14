import numpy as np

class AdversarialGenerator:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
    
    def generate_adversarial_noise(self, data):
        """Add targeted noise to make normal patterns look anomalous"""
        noise = np.random.normal(0, self.epsilon, data.shape)
        noise = noise + (self.epsilon * np.sign(noise))
        return data + noise 
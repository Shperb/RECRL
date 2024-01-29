import numpy as np
import math


class ExponentialDecay():  #Algorithm 1: lines 5-6
    def compute_cost(self, epoch, epochs,decay_end,d_t,d_d):
        progress = min(1, epoch / np.floor(decay_end * epochs - 1))
        current_cost_lim = (d_t - d_d)/(1-math.exp(-1))*math.exp(-progress) + (d_d - math.exp(-1)*d_t)/(1-math.exp(-1))
        return current_cost_lim

class LinearDecay(): #Algorithm 1: lines 7-8
    def compute_cost(self, epoch, epochs,decay_end,d_t,d_d):
        progress = min(1, epoch / np.floor(decay_end * epochs - 1))
        current_cost_lim = d_d * (
                    1 + (d_t - 1) * (1 - progress))
        return current_cost_lim


class CosineDecay(): #Algorithm 1: lines 9-10
    def compute_cost(self, epoch, epochs,decay_end,d_t,d_d):
        progress = min(1, epoch / np.floor(decay_end * epochs - 1))
        cosine_of_progress = np.cos(progress * np.pi/2)
        current_cost_lim = d_d * (
                    1 + (d_t - 1) * cosine_of_progress)
        return current_cost_lim

class NoDecay():
    def compute_cost(self, epoch, epochs,decay_end,d_t,d_d):
        return d_d
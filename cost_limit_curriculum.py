from copy import deepcopy
import numpy as np

class StaticCostLimitCurriculum: #Algorithm 1
    def __init__(self, d_t, cost_decay, **kwargs):
        self.d_t = d_t
        self.cost_decay = cost_decay
        self.params = deepcopy(kwargs)

    def compute_cost_and_noise(self, epoch, epochs, costs, returns):
        return self.cost_decay.compute_cost(epoch, epochs,self.params.get('decay_end'),self.params.get('d_t'),self.d_t),0

    def get_initial_cost(self):
        return self.d_t




class DynamicCurriculum: #Algorithm 2
    def __init__(self, d_t, **kwargs):
        self.d_t = d_t
        self.params = deepcopy(kwargs)
        self.window_counter = 0
        self.window = self.params.get('window')
        self.d = self.get_initial_cost()
        self.previous_epoc_ranges = [0]
        self.window_start =0

    def compute_cost_and_noise(self, epoch, epochs, costs, returns):
        self.window_counter += 1
        noise = 0
        if epoch < self.window*2:
            self.previous_epoc_ranges.append(len(costs))
            return self.get_initial_cost(),noise

        cost = min(self.d, np.mean(costs[self.previous_epoc_ranges[-self.window]:]))
        convergence_epoch = epochs * self.params.get('decay_end') - 1
        if epoch >= convergence_epoch or cost <= self.d_t:
            self.d = self.d_t
        else:
            cost_diff = abs(np.mean(costs[self.previous_epoc_ranges[-self.window]:]) - np.mean(costs[self.previous_epoc_ranges[-2*self.window]:self.previous_epoc_ranges[-self.window]]))
            cost_normalizer = np.mean(costs[self.previous_epoc_ranges[-2*self.window]:self.previous_epoc_ranges[-self.window]])
            normalized_cost = cost_diff / cost_normalizer #difference in cost
            return_diff = abs(np.mean(returns[self.previous_epoc_ranges[-self.window]:]) - np.mean(returns[self.previous_epoc_ranges[-2*self.window]:self.previous_epoc_ranges[-self.window]]))
            return_normalizer = np.mean(returns[self.previous_epoc_ranges[-2*self.window]:self.previous_epoc_ranges[-self.window]])
            normalized_return = return_diff / return_normalizer #difference in reward
            if self.window_counter > self.window and normalized_cost < self.params.get('cost_delta_threshold') and normalized_return < self.params.get('return_delta_threshold'): #line 5
                if np.mean(costs[self.previous_epoc_ranges[-self.window]:]) > self.d:
                    epochs_before_convergence = convergence_epoch - epoch
                    self.d = cost - (cost - self.d_t)/(epochs_before_convergence/self.window) #line 7
                else:
                    noise = self.params.get('initial_noise') #line 9
            self.window_counter = 0
            self.window_start = epoch
        self.previous_epoc_ranges.append(len(costs))
        return self.d, noise


    def get_initial_cost(self):
        return self.d_t

from copy import deepcopy

class HardNegativeMiner:
    def __init__(self, rate):
        self.rate = rate

        self.cache = None
        self.worst_loss = 0
        self.idx = 0

    def update_cache(self, meter, data):
        loss = meter['loss']
        if loss > self.worst_loss:
            self.cache = deepcopy(data)
            self.worst_loss = loss
        self.idx += 1

    def need_iter(self):
        return self.idx >= self.rate

    def invalidate_cache(self):
        self.worst_loss = 0
        self.cache = None
        self.idx = 0
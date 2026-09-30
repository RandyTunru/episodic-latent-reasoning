import math

class WDScheduler:
    def __init__(self, config: dict):
        self.config = config
        self.scheduler = self._get_wd

    def _get_wd(self, step: int) -> float:
        """Weight decay ramped up from weight_decay to weight_decay_end on the
        lr's cosine shape (paper: 0.04 -> 0.4; the reference code schedules wd
        with the same cosine). None when no weight_decay_end is configured.
        
        If weight_decay_end is None, the weight decay remains constant at weight_decay.
        """
        start = self.config.get('weight_decay')
        end = start if self.config.get('weight_decay_end') is None else self.config.get('weight_decay_end')
        if start is None or end is None:
            return None
        progress = min(step / self.config['max_steps'], 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 at step 0 -> 0 at max_steps

        # If end > start, this will ramp up from start to end; 
        # if end < start, this will ramp down from start to end. 
        # if end == start (incl if only weight_decay is specified), this will remain constant at the weight_decay value.
        return end - (end - start) * coeff 
    
    def get_weight_decay(self, step: int) -> float:
        return self.scheduler(step)
import math

class LRScheduler:
    def __init__(self, config: dict):
        self.config = config
        self.scheduler = self._get_cosine_lr 

    def _get_cosine_lr(self, step: int) -> float:
        """Cosine learning-rate schedule with linear warmup."""
        max_lr = self.config["learning_rate"]
        min_lr = max_lr * self.config.get("min_lr_alpha", 0.1)
        warmup_steps = self.config.get("warmup_steps", 1000)
        total_steps = self.config["max_steps"]

        # Setting to 1 means standard cosine annealing, >1 means multiple cycles of cosine annealing
        # max(1, ...) ensures that we always have at least 1 cycle, even if the config is set to 0 or negative.
        num_cycles = max(self.config.get("learning_rate_cycle", 1), 1) 

        # Linear warmup
        if step < warmup_steps:
            return max_lr * (step / max(warmup_steps, 1))

        # Beyond total steps
        if step >= total_steps:
            return min_lr

        # Cosine decay with cycles
        cycle_length = (total_steps - warmup_steps) / num_cycles # Equally divide the non warmup steps into cycles
        cycle_step = (step - warmup_steps) % cycle_length # Where in the current cycle we are

        decay_ratio = cycle_step / max(cycle_length, 1) # Ratio of how far we are in the current cycle
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Cosine decay coefficient for the current cycle
        return min_lr + coeff * (max_lr - min_lr)


    # Legacy code for cosine learning with restarts, but with increasing intervals.
    # This is not currently used, but kept for reference.
    # The original code was shaped with the idea of having this as a mode
    # Hence the use of class with an exposed get_lr method.

    # def _get_cosine_restart_lr(self, step: int) -> float:
    #     """Cosine learning-rate schedule with linear warmup and warm restarts."""
    #     max_lr = self.config["learning_rate"]
    #     min_lr = max_lr * self.config.get("min_lr_alpha", 0.1)
    #     warmup_steps = self.config.get("warmup_steps", 1000)
    #     total_steps = self.config["max_steps"]
    #     restart_interval = self.config.get("restart_interval", 10000)
    #     interval_multiplier = self.config.get("restart_interval_multiplier", 1.0)

    #     # Linear warmup
    #     if step < warmup_steps:
    #         return max_lr * (step / max(warmup_steps, 1))

    #     # Beyond total steps; more of an edge case since we should be stopping training at max_steps, but just in case.
    #     if step >= total_steps:
    #         return min_lr 

    #     # Determine start/end steps for current restart cycle
    #     cycle_start_step = warmup_steps
    #     current_interval = float(restart_interval)

    #     if interval_multiplier != 1.0:
    #         while step >= cycle_start_step + current_interval:
    #             cycle_start_step += current_interval
    #             current_interval *= interval_multiplier
    #     else:
    #         cycle = (step - warmup_steps) // restart_interval
    #         cycle_start_step = warmup_steps + cycle * restart_interval

    #     cycle_end_step = cycle_start_step + current_interval
    #     # Squeeze the final partial cycle so the schedule always ends at
    #     # min_lr exactly at total_steps, instead of stopping mid-cycle at
    #     # a high LR when the intervals don't tile max_steps evenly.
    #     if cycle_end_step > total_steps:
    #         cycle_end_step = float(total_steps)

    #     # Cosine decay within the active cycle
    #     decay_ratio = (step - cycle_start_step) / max(cycle_end_step - cycle_start_step, 1)
    #     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    #     return min_lr + coeff * (max_lr - min_lr)

    def get_lr(self, step: int) -> float:
        return self.scheduler(step)
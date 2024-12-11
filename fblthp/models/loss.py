def compute_beta(step, beta_start, beta_end, warmup_steps):
    """
    Compute the KL weight (beta) based on the current step.
    Args:
        step (int): Current training step.
        beta_start (float): Initial beta value.
        beta_end (float): Final beta value.
        warmup_steps (int): Number of steps for annealing.
    Returns:
        float: Annealed beta value.
    """
    if step >= warmup_steps:
        return beta_end
    return beta_start + (beta_end - beta_start) * (step / warmup_steps)


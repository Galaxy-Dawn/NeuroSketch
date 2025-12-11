import torch


def random_shift(X, shift_ratio=0.2, method="bi"):
    max_shift = int(X.size(-1) * shift_ratio)
    X_shifted = torch.zeros_like(X)
    if method == "bi":
        n_steps = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        if n_steps > 0:
            X_shifted[..., n_steps:] = X[..., :-n_steps]
        elif n_steps < 0:
            X_shifted[..., :n_steps] = X[..., -n_steps:]
        return X_shifted
    elif method == "forward":
        n_steps = torch.randint(0, max_shift + 1, (1,)).item()
        X_shifted[..., :-n_steps] = X[..., n_steps:]
        return X_shifted
    elif method == "backward":
        n_steps = torch.randint(0, max_shift + 1, (1,)).item()
        X_shifted[..., n_steps:] = X[..., :-n_steps]
        return X_shifted
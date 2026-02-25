import numpy as np
from scipy.optimize import root

def calc_spiral_arc_length(theta, a):
    """Calculates arc length L of spiral r = a * theta from 0 to theta."""
    if theta == 0:
        return 0.0
    # Exact formula for Archimedean spiral arc length
    return (a / 2.0) * (theta * np.sqrt(1 + theta**2) + np.log(theta + np.sqrt(1 + theta**2)))

def find_theta_from_length(target_L, a, initial_guess):
    """Numerically solves for theta given a target arc length."""
    # Define the function whose root we want to find: f(theta) - target_L = 0
    func = lambda t: calc_spiral_arc_length(t, a) - target_L

    # root() expects the function to accept and return array-like values
    result = root(func, initial_guess)

    if not result.success:
        raise RuntimeError(f"Root finding failed: {result.message}")

    return result.x[0]

# --- Example Usage ---
a = 1.5           # Growth constant (r = a * theta)
theta_initial = 2 * np.pi  # Starting angle (1 full turn)
delta_L = 10.0    # Desired change in arc length

# 1. Calculate current length
L_initial = calc_spiral_arc_length(theta_initial, a)

# 2. Calculate target length
L_target = L_initial + delta_L

# 3. Solve for new theta (use theta_initial as a starting guess for faster convergence)
theta_new = find_theta_from_length(L_target, a, theta_initial)

# 4. Calculate change in theta
delta_theta = theta_new - theta_initial

print(f"Initial Theta: {theta_initial:.4f} rad")
print(f"New Theta:     {theta_new:.4f} rad")
print(f"Change (Δθ):   {delta_theta:.4f} rad")

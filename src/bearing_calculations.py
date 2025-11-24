import numpy as np


def NewtonDescent(S_s, init_value=0.5, MAX_ITER=5000, tol=10**-10):
    def f(epsilon):
        return (
            epsilon**8
            - 4 * epsilon**6
            + (6 - (S_s**2) * (16 - np.pi**2)) * epsilon**4
            - (4 + (np.pi**2) * (S_s**2)) * epsilon**2
            + 1
        )

    def F(epsilon):
        return (
            8 * epsilon**7
            - 24 * epsilon**5
            + 4 * (6 - (S_s**2) * (16 - np.pi**2)) * epsilon**3
            - 2 * (4 + (np.pi**2) * (S_s**2)) * epsilon
        )

    iter = 0
    epsilon = init_value
    while abs(f(epsilon)) > tol and iter < MAX_ITER:
        if F(epsilon) != 0:
            epsilon -= f(epsilon) / F(epsilon)
        else:
            epsilon -= f(epsilon) / (F(epsilon) + 10**-15)
        iter += 1
        if iter == MAX_ITER:
            print(f"Max iteration reached, error = {f(epsilon)}")
    return epsilon


def solve_eccentricity(D, omega, eta, L, f, c):
    """
    Solve for eccentricity (epsilon) based on the provided parameters.
    Parameters:
        D (float): Diameter (m)
        omega (float): Shaft speed (rad/s)
        eta (float): Viscosity (Pa.s)
        L (float): Length (m)
        f (float): Static force/load (N)
        c (float): Radial clearance (m)
    Returns:
        epsilon (float): The solved eccentricity.
    """

    for initial_guess in [0.5, 0.25, 0.75]:
        S_s = D * omega * eta * L**3 / (8 * f * c**2)
        print("S_s:", S_s)  # DEBUG
        epsilon = NewtonDescent(S_s, init_value=initial_guess)

        # Check if epsilon is within valid range
        if epsilon > 0 and epsilon <= 1:
            return epsilon

    # Newton descent did not converge
    return None


def solve_K_and_C(D, omega, eta, L, f, c):
    """
    Solve for constants C and K based on the provided parameters.
    Parameters:
        D (float): Diameter (m)
        omega (float): Shaft speed (rad/s)
        eta (float): Viscosity (Pa.s)
        L (float): Length (m)
        f (float): Static force/load (N)
        c (float): Radial clearance (m)
    Returns:
    K (float): The solved stiffness matrix K
    C (float): The solved damping matrix C
    epsilon (float): The solved eccentricity.
    """

    # Solve epsilon first
    ##
    epsilon = solve_eccentricity(D, omega, eta, L, f, c)
    print("Eccentricity (epsilon):", epsilon)  # DEBUG

    # Solve matrix subparts
    ##

    # Needed for all
    h_0 = 1 / (np.pi**2 * (1 - epsilon**2) + 16 * epsilon**2) ** (3 / 4)

    #
    a_uu = h_0 * 4 * (np.pi**2 * (2 - epsilon**2) + 16 * epsilon**2)

    #
    a_uv = (
        h_0
        * np.pi
        * (np.pi**2 * (1 - epsilon**2) - 16 * epsilon**4)
        / (epsilon * np.sqrt(1 - epsilon**2))
    )

    #
    a_vu = (
        -h_0
        * np.pi
        * (
            np.pi**2 * (1 - epsilon**2) * (1 + 2 * epsilon**2)
            + 32 * epsilon**2 * (1 + epsilon**2)
        )
        / (epsilon * np.sqrt(1 - epsilon**2))
    )

    #
    a_vv = (
        h_0
        * 4
        * (
            np.pi**2 * (1 + 2 * epsilon**2)
            + (32 * epsilon**2 * (1 + epsilon**2) / (1 - epsilon**2))
        )
    )

    #
    b_uu = (
        h_0
        * (
            2
            * np.pi
            * np.sqrt(1 - epsilon**2)
            * (np.pi**2 * (1 + 2 * epsilon**2) - 16 * epsilon**2)
        )
        / epsilon
    )

    #
    b_vu = b_uv = -h_0 * 8 * (np.pi**2 * (1 + 2 * epsilon**2) - 16 * epsilon**2)

    #
    b_vv = (
        h_0
        * (2 * np.pi * (np.pi**2 * (1 - epsilon**2) + 48 * epsilon**2))
        / (epsilon * np.sqrt(1 - epsilon**2))
    )

    # Build C and K
    ##

    K = f / c * np.array([[a_uu, a_uv], [a_vu, a_vv]])
    C = f / (c * omega) * np.array([[b_uu, b_uv], [b_vu, b_vv]])

    return K, C

def solve_eccentricity(D, omega, eta, L, f, c):
    """
    Solve for eccentricity (epsilon) based on the provided parameters.
    Parameters:
        D (float): A given parameter.
        omega (float): A given parameter.
        eta (float): Viscosity (Pa.s)
        L (float): Length (mm)
        f (float): Static force/load (N)
        c (float): Radial clearance (mm)
    Returns:
        epsilon (float): The solved eccentricity.
    """
    epsilon = None

    # TODO

    return epsilon

def solve_C_and_K(D, omega, eta, L, f, c):
    """
    Solve for constants C and K based on the provided parameters.    
    Parameters:
        D (float): A given parameter.
        omega (float): A given parameter.
        eta (float): Viscosity (Pa.s)
        L (float): Length (mm)
        f (float): Static force/load (N)
        c (float): Radial clearance (mm)
    Returns:
    C (float): The solved damping matrix C.
    K (float): The solved stiffness matrix K.
    """
    C = None
    K = None

    epsilon = solve_eccentricity(D, omega, eta, L, f, c)

    # TODO

    return C, K

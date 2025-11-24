import numpy as np

# Friswell specific functions
#############################


def NewtonDescent(S_s, init_value=0.5, MAX_ITER=5000, tol=10**-10):
    """
    Find a root of the Friswell polynomial using Newton's method.

    Parameters
    ----------
    S_s : float
        Dimensionless parameter S_s = D * omega * eta * L**3 / (8 * f * c**2).
    init_value : float, optional
        Initial guess for epsilon (default 0.5).
    MAX_ITER : int, optional
        Maximum allowed Newton iterations (default 5000).
    tol : float, optional
        Absolute tolerance on the polynomial residual |f(epsilon)| used as
        the convergence criterion (default 1e-10).

    Returns
    -------
    float
        Approximated root epsilon (dimensionless eccentricity).

    Raises
    ------
    RuntimeError
        If the maximum number of iterations is reached without convergence.
    """

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
            raise RuntimeError(f"Max iteration reached, error = {f(epsilon)}")
    return epsilon


def solve_eccentricity(D, omega, eta, L, f, c):
    """
    Solve for eccentricity (epsilon) using NewtonDescent.

    Parameters
    ----------
    D : float
        Shaft diameter (m).
    omega : float
        Shaft speed (rad/s).
    eta : float
        Fluid viscosity (Pa·s).
    L : float
        Bearing length (m).
    f : float
        Static radial load (N).
    c : float
        Radial clearance (m).

    Returns
    -------
    float or None
        Eccentricity (dimensionless, 0 < epsilon <= 1) if a valid solution is found;
        otherwise None.
    """
    for initial_guess in [0.5, 0.25, 0.75]:
        S_s = D * omega * eta * L**3 / (8 * f * c**2)
        # print("DEBUG - S_s:", S_s)  # DEBUG
        epsilon = NewtonDescent(S_s, init_value=initial_guess)

        # Check if epsilon is within valid range
        if epsilon > 0 and epsilon <= 1:
            return epsilon

    # Newton descent did not converge
    return None


def solve_K_and_C_Friswell(omega, f, c, epsilon):
    """
    Compute the 2x2 stiffness (K) and damping (C) matrices for a fluid bearing.

    Parameters
    ----------
    omega : float
        Shaft speed (rad/s).
    f : float
        Static force / load (N).
    c : float
        Radial clearance (m).
    epsilon : float
        Eccentricity (dimensionless, 0 < epsilon <= 1).

    Returns
    -------
    K : numpy.ndarray, shape (2, 2)
        Stiffness matrix (N/m), computed as (f / c) * A where A is dimensionless.
    C : numpy.ndarray, shape (2, 2)
        Damping matrix (N·s/m), computed as (f / (c * omega)) * B where B is dimensionless.
    """
    # Solve matrix subparts
    ##

    # Needed for all
    h_0 = np.pi**2 * (1 - epsilon**2)
    h_0 += 16 * epsilon**2
    h_0 = h_0 ** (3 / 2)
    h_0 = 1 / h_0

    #
    a_uu = h_0 * 4 * (np.pi**2 * (2 - epsilon**2) + 16 * epsilon**2)

    #
    a_uv = (
        h_0
        * np.pi
        * (np.pi**2 * (1 - epsilon**2) ** 2 - 16 * epsilon**4)
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
            + ((32 * epsilon**2 * (1 + epsilon**2)) / (1 - epsilon**2))
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
        * (2 * np.pi * (np.pi**2 * (1 - epsilon**2) ** 2 + 48 * epsilon**2))
        / (epsilon * np.sqrt(1 - epsilon**2))
    )

    # Build C and K
    ##

    K = f / c * np.array([[a_uu, a_uv], [a_vu, a_vv]])
    C = f / (c * omega) * np.array([[b_uu, b_uv], [b_vu, b_vv]])

    return K, C


# Al-Bender specific function
#############################


def compute_Fbar(Lambda, L_over_D):
    """
    Compute the complex function Fbar(Λ; L/D) = Fr + i·Ft used in Al-Bender formulations (chapter 9.3).

    Parameters
    ----------
    Lambda : scalar or array-like
        Non-negative real or complex-valued frequency parameter Λ. Can be a scalar or array and will be
        converted to a complex numpy array for vectorized evaluation.
    L_over_D : float
        The ratio L/D.

    Returns
    -------
    numpy.ndarray (complex128)
        Complex-valued array (Fr + 1j*Ft) evaluated elementwise.
    """

    lam = np.array(Lambda, dtype=np.complex128)
    s = np.sqrt(1.0 + 1j * lam)
    arg = s * L_over_D

    term = 1.0 - np.tanh(arg) / arg
    pref = (1j * lam) / (1.0 + 1j * lam)
    Fbar = pref * term * (np.pi / 2.0)

    return Fbar


def solve_K_and_C_AlBender(Lambda, sigma_array, L_over_D=1.0):
    """
    Compute dynamic stiffness and damping matrices K and C using Al-Bender's method for some Λ, σ, and L/D. Be careful with very small values of σ.

    Parameters
    ----------
    Lambda : scalar or array-like
        Frequency parameter Λ. May be a scalar or an array; it will be broadcast
        against sigma_array.
    sigma_array : array-like
        Array of positive real offsets (σ). Must be non-zero (avoid values
        near zero). Must be broadcastable to the shape of Lambda.
    L_over_D : float, optional
        Geometry ratio L/D passed to compute_Fbar (default 1.0).

    Returns
    -------
    KXX, KXY, CXX, CXY : numpy.ndarray
        Arrays (or scalars) of the same broadcasted shape as the inputs:
        - K_xx: dynamic stiffness component (Fr averaged)
        - K_xy: dynamic coupling component (Ft averaged)
        - C_xx: dynamic damping component corresponding to K_xx
        - C_xy: dynamic damping coupling corresponding to K_xy
    """

    Lam_minus = Lambda - sigma_array
    Lam_plus = Lambda + sigma_array

    F_minus = compute_Fbar(Lam_minus, L_over_D)
    F_plus = compute_Fbar(Lam_plus, L_over_D)

    Fr_minus = np.real(F_minus)
    Ft_minus = np.imag(F_minus)
    Fr_plus = np.real(F_plus)
    Ft_plus = np.imag(F_plus)

    # Stiffness
    K_xx = 0.5 * (Fr_minus + Fr_plus)
    K_yx = 0.5 * (Ft_minus + Ft_plus)

    # Damping
    C_xx = -0.5 / sigma_array * (Ft_minus - Ft_plus)
    C_yx = -0.5 / sigma_array * (Fr_minus - Fr_plus)

    # Build K and C matrices
    K = np.array(
        [
            [K_xx, -K_yx],
            [K_yx, K_xx],
        ]
    )
    C = np.array(
        [
            [C_xx, -C_yx],
            [C_yx, C_xx],
        ]
    )

    return K, C

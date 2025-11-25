import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def compressible_flow(
    mu=0.04,
    omega=2 * np.pi,
    r_in=0.05,
    r_out=0.10,
    n_sector=6,
    Delta_theta=np.pi / 4,
    hL=50e-6,
    hT=10e-6,
    boundary=False,
    rho_a=1.2,
    p_a=101325,
    beta=1e6,  # 2e6
    max_iter=80,
    tol=1e-6,
    relax=0.4,
    plot_title=True,
    savefig=False,
    out_figure="compressible_flow.png",
):
    """
    Illustrate the pressure distribution of a tilted pad thrust
    bearing model when the opposing collar rotates counter-
    clockwise with angular velocity ω, and the fluid is compressible.

    Parameters
    ----------
    mu : float
        dynamic viscosity [Pa.s]
    omega : float
        angular speed [rad/s]
    r_in : float
        inner radius [m]
    r_out : float
        outer radius [m]
    n_sector : int
        number of sectors
    Delta_theta : float
        angular width of each sector [rad],
        should be smaller than np.pi/n_sector
    hL : float
        leading edge gap height [m]
    hT : float
        trailing edge gap height [m]
    boundary : bool
        include Dirichlet boundary condition or not
    rho_a : float
        reference density [kg/m^3]
    p_a : float
        ambient pressure [Pa]
    beta : float
        compressibility parameter [Pa]
    max_iter : int
        maximum number of iterations
    tol : float
        tolerance on infinity norm of pressure update
    relax : float
        under-relaxation parameter (0 < relax <= 1)
    plot_title : bool
        include titles in plots or not
    out_figure : str
        file name to save plots
    """
    Nr = 40
    Ntheta = 60

    r = np.linspace(r_in, r_out, Nr)
    theta = np.linspace(0.0, Delta_theta, Ntheta)
    dr = r[1] - r[0]
    dtheta = theta[1] - theta[0]
    L = r_out * Delta_theta

    h_r_theta = lambda r, theta: hT + (hL - hT) * (r * theta / L)
    idx = lambda i, j: i * Ntheta + j

    TH, RR = np.zeros(Ntheta+2), np.zeros(Nr+2)
    TH[1:-1], RR[1:-1] = theta, r
    TH[0], TH[-1], = -dtheta, Delta_theta+dtheta
    RR[0], RR[-1] = r_in-dr, r_out+dr
    TH, RR = np.meshgrid(TH, RR)
    H = h_r_theta(RR, TH)
    H3 = H**3  # Precompute H^3

    coef_const = 1 / (12 * mu)
    dh_dtheta = lambda radius: (hL - hT) * radius / (r_out * Delta_theta)
    rho_of_p = lambda p: rho_a * np.exp((p - p_a) / beta)

    N = Nr * Ntheta
    p_vec = np.zeros(N)

    # Iterative (Picard) loop: assemble A(p) and b(p), solve A p_new = b
    start_time = time.time()
    converged = False
    for it in range(1, max_iter + 1):
        rows = []
        cols = []
        vals = []
        b = np.zeros(N)
        rho_all = rho_of_p(p_vec)

        for i in range(1, Nr+1):
            ri = r[i-1]
            for j in range(1, Ntheta+1):
                k = idx(i-1, j-1)
                
                if boundary:
                    if j == 1 or j == Ntheta:
                        rows.append(k)
                        cols.append(k)
                        vals.append(1.0)
                        b[k] = 0.0
                        continue
                    
                    if i == 1 or i == Nr:
                        rows.append(k)
                        cols.append(k)
                        vals.append(1.0)
                        b[k] = 0.0
                        continue

                # Theta-term coefficients
                Hm = H3[i, j - 1]
                Hp = H3[i, j + 1]
                A_m = coef_const * Hm / ri
                A_p = coef_const * Hp / ri
                theta_m_coef = -A_m / (dtheta * dtheta)
                theta_p_coef = -A_p / (dtheta * dtheta)
                theta_c_coef = -(theta_m_coef + theta_p_coef)

                # Radial-term coefficients
                H_im = H3[i - 1, j]
                H_ip = H3[i + 1, j]
                if i>=2: 
                    B_im = coef_const * H_im * r[i-2]
                else:
                    B_im = 0
                if i <= Nr-1:
                    B_ip = coef_const * H_ip * r[i]
                else:
                    B_ip = 0
                rad_m_coef = -B_im / (dr * dr)
                rad_p_coef = -B_ip / (dr * dr)
                rad_c_coef = -(rad_m_coef + rad_p_coef)

                center = theta_c_coef + rad_c_coef
                coef_jm = theta_m_coef
                coef_jp = theta_p_coef
                coef_im = rad_m_coef
                coef_ip = rad_p_coef

                # Add to sparse lists
                rows.append(k)
                cols.append(k)
                vals.append(center)
                if j > 1:
                    rows.append(k)
                    cols.append(idx(i-1, j-2)) #(i,j-1)
                    vals.append(coef_jm)
                if j < Ntheta:
                    rows.append(k)
                    cols.append(idx(i-1, j)) #(i,j+1)
                    vals.append(coef_jp)
                if i > 1:
                    rows.append(k)
                    cols.append(idx(i-2, j-1)) #(i-1,j)
                    vals.append(coef_im)
                if i < Nr:
                    rows.append(k)
                    cols.append(idx(i, j-1)) #(i+1,j)
                    vals.append(coef_ip)

                # RHS source term: (omega * r / 2) * d/dtheta( rho * h )
                # d( rho * h )/dtheta = rho * dh/dtheta + h * d rho/dtheta
                # approximate d rho / dtheta
                rho_c = rho_all[k]
                rho_m = rho_all[idx(i-1, j-2)] #(i, j-1)
                drho_dtheta = (rho_c - rho_m) / dtheta

                dhdt = dh_dtheta(ri)

                b[k] = (omega * ri / 2.0) * (rho_c * dhdt + H[i, j] * drho_dtheta)

        A = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()

        # Solve (handle possible singular/ill conditioned by catching errors)
        try:
            p_new = spsolve(A, b)
        except Exception as e:
            print("Linear solver failed at iteration", it, "with error:", e)
            break

        # enforce Dirichlet at theta boundaries explicitly (should already be set)
        for i_dir in range(Nr):
            p_new[idx(i_dir, 0)] = 0.0
            p_new[idx(i_dir, Ntheta - 1)] = 0.0

        # check convergence
        diff = p_new - p_vec
        err = np.linalg.norm(diff, np.inf)
        print(f"Iter {it:3d}: ||p_new - p||_inf = {err:.3e}")

        if err < tol:
            p_vec = p_new.copy()
            converged = True
            print(f"Converged in {it} iterations (tol={tol}).")
            break

        p_vec = relax * p_new + (1.0 - relax) * p_vec

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} s")
    if not converged:
        print(
            "Warning: solver did not converge within max_iter. Consider increasing iterations, decreasing relax or changing tol."
        )

    # reshape for plotting
    P = p_vec.reshape((Nr, Ntheta))
    P_clip = np.abs(P)

    # Plot and save results
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    for i in range(n_sector):
        TH, RR = np.meshgrid(theta + i * 2 * np.pi / n_sector, r)
        X = RR * np.cos(TH)
        Y = RR * np.sin(TH)
        cf = ax1.contourf(
            X,
            Y,
            P_clip,
            vmin=np.min(P_clip),
            vmax=np.max(P_clip),
            cmap="viridis",
            levels=50,
        )
    ax1.set(xticks=[], yticks=[])
    if plot_title:
        ax1.set_title(rf"Pressure dist. when compressibility $\beta$ = {beta:.1e} Pa")
    ax1.set_aspect("equal", "box"),
    ax1.spines[["left", "bottom", "top", "right"]].set_visible(False)
    plt.colorbar(cf, ax=ax1, orientation="vertical")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(theta, P_clip[0, :], "--", label=f"r={r_in:.3f} m", color="C1")
    ax2.plot(
        theta, P_clip[Nr // 2, :], "-.", label=f"r={(r_in+r_out)/2:.3f} m", color="C2"
    )
    ax2.plot(theta, P_clip[-1, :], ":", label=f"r={r_out:.3f} m", color="C3")
    ax2.set_xlabel("theta (rad)")
    ax2.set_ylabel("pressure (Pa)")
    ax2.legend()
    if plot_title:
        ax2.set_title("Pressure vs theta at different radius")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if savefig:
        fig.savefig(os.path.join("reports", out_figure), dpi=200)
        print(f"Figure saved to: {os.path.join('reports', out_figure)}")

    # Print simple diagnostics
    print("Peak pressure (abs) [Pa]:", np.max(P_clip))
    print("Mean pressure (abs) [Pa]:", np.mean(P_clip))

    plt.show()


if __name__ == "__main__":
    compressible_flow(beta=1e7, out_figure="compressible_flow.png", savefig=True)
    compressible_flow(beta=4e6, out_figure="compressible_flow_2.png", savefig=True)
    compressible_flow(beta=1e7, boundary=True, out_figure="compressible_flow_w_boundary.png", savefig=True)
    compressible_flow(beta=4e6, boundary=True, out_figure="compressible_flow_2_w_boundary.png", savefig=True)
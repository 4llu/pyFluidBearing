import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time

# ---------------------------
# Parameters (user-editable)
# ---------------------------
mu = 0.04             # dynamic viscosity [Pa.s]
omega = 2*np.pi       # angular speed [rad/s]
r_in = 0.05           # inner radius [m]
r_out = 0.10          # outer radius [m]
Delta_theta = np.pi/4 # sector angular width [rad]

hL = 50e-6
hT = 10e-6

# compressibility parameters
rho_a = 1.2           # reference density (kg/m^3) - keep in consistent units
p_a = 101325          # ambient pressure (Pa)
beta = 1e6            # compressibility parameter [Pa] (example)

# numerical grid
Nr = 40
Ntheta = 40

out_figure = "compressible_flow.png"

def h_r_theta(r, theta):
    s = r * theta
    L = r_out * Delta_theta
    return hL + (hT - hL) * (s / L)

r = np.linspace(r_in, r_out, Nr)
theta = np.linspace(0.0, Delta_theta, Ntheta)
dr = r[1] - r[0]
dtheta = theta[1] - theta[0]

# flatten helper
def idx(i, j):
    return i * Ntheta + j

# Precompute h and H^3 on grid (h itself used in RHS)
TH, RR = np.meshgrid(theta, r)
H = h_r_theta(RR, TH)
H3 = H**3

# derivative dh/dtheta (analytical for linear tilt)
dh_dtheta = lambda radius: (hT - hL) * radius / (r_out * Delta_theta)

# density law
def rho_of_p(p):
    # returns rho for scalar p or numpy array
    return rho_a * np.exp((p - p_a) / beta)

# ------------------------------------------------------------------
# Nonlinear solver settings
# ------------------------------------------------------------------
coef_const = 1.0 / (12.0 * mu)

N = Nr * Ntheta

max_iter = 80
tol = 1e-6            # tolerance on infinity norm of pressure update
relax = 0.4           # under-relaxation parameter (0 < relax <= 1)

# initial guess: zero pressure
p_vec = np.zeros(N)

# helper to build sparse matrix from lists quickly
def build_sparse(rows, cols, vals):
    return sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()

# ------------------------------------------------------------------
# Iterative (Picard) loop: assemble A(p) and b(p), solve A p_new = b
# ------------------------------------------------------------------
start_time = time.time()
converged = False

for it in range(1, max_iter + 1):
    rows = []
    cols = []
    vals = []
    b = np.zeros(N)

    # precompute rho at all nodes from current p_vec
    rho_all = rho_of_p(p_vec)  # length N

    for i in range(Nr):
        ri = r[i]
        for j in range(Ntheta):
            k = idx(i, j)

            # Theta boundaries: Dirichlet p=0
            if j == 0 or j == Ntheta-1:
                rows.append(k); cols.append(k); vals.append(1.0)
                b[k] = 0.0
                continue

            # Theta-term coefficients (use rho * H^3)
            Hc_rho = rho_all[k] * H3[i, j]
            Hm_rho = rho_all[idx(i, j-1)] * H3[i, j-1]
            Hp_rho = rho_all[idx(i, j+1)] * H3[i, j+1]

            A_m = coef_const * Hm_rho / ri
            A_p = coef_const * Hp_rho / ri

            theta_m_coef = -A_m / (dtheta * dtheta)
            theta_p_coef = -A_p / (dtheta * dtheta)
            theta_c_coef = - (theta_m_coef + theta_p_coef)

            # Radial-term coefficients: d/dr( (rho*H^3/(12 mu)) * r * dp/dr )
            if 0 < i < Nr-1:
                H_im_rho = rho_all[idx(i-1, j)] * H3[i-1, j]
                H_ip_rho = rho_all[idx(i+1, j)] * H3[i+1, j]
                B_im = coef_const * H_im_rho * r[i-1]
                B_ip = coef_const * H_ip_rho * r[i+1]
                rad_m_coef = -B_im / (dr * dr)
                rad_p_coef = -B_ip / (dr * dr)
                rad_c_coef = - (rad_m_coef + rad_p_coef)
            else:
                # Neumann dp/dr = 0 approximated with ghost point
                if i == 0:
                    H_ip_rho = rho_all[idx(i+1, j)] * H3[i+1, j]
                    B_ip = coef_const * H_ip_rho * r[i+1]
                    rad_m_coef = 0.0
                    rad_p_coef = -B_ip / (dr * dr)
                    rad_c_coef = -rad_p_coef
                else:  # i == Nr-1
                    H_im_rho = rho_all[idx(i-1, j)] * H3[i-1, j]
                    B_im = coef_const * H_im_rho * r[i-1]
                    rad_p_coef = 0.0
                    rad_m_coef = -B_im / (dr * dr)
                    rad_c_coef = -rad_m_coef

            center = theta_c_coef + rad_c_coef
            coef_jm = theta_m_coef
            coef_jp = theta_p_coef
            coef_im = rad_m_coef
            coef_ip = rad_p_coef

            # Add to sparse lists
            rows.append(k); cols.append(k); vals.append(center)
            rows.append(k); cols.append(idx(i, j-1)); vals.append(coef_jm)
            rows.append(k); cols.append(idx(i, j+1)); vals.append(coef_jp)
            if i-1 >= 0:
                rows.append(k); cols.append(idx(i-1, j)); vals.append(coef_im)
            if i+1 < Nr:
                rows.append(k); cols.append(idx(i+1, j)); vals.append(coef_ip)

            # RHS source term: (omega * r / 2) * d/dtheta( rho * h )
            # d( rho * h )/dtheta = rho * dh/dtheta + h * d rho/dtheta
            # approximate d rho / dtheta with central difference
            rho_c = rho_all[k]
            rho_m = rho_all[idx(i, j-1)]
            rho_p = rho_all[idx(i, j+1)]
            drho_dtheta = (rho_p - rho_m) / (2.0 * dtheta)

            dhdt = dh_dtheta(ri)

            b[k] = (omega * ri / 2.0) * (rho_c * dhdt + H[i, j] * drho_dtheta)

    # finalize sparse matrix and solve linear system for p_new
    A = build_sparse(rows, cols, vals)

    # Solve (handle possible singular/ill conditioned by catching errors)
    try:
        p_new = spsolve(A, b)
    except Exception as e:
        print("Linear solver failed at iteration", it, "with error:", e)
        break

    # enforce Dirichlet at theta boundaries explicitly (should already be set)
    for i_dir in range(Nr):
        p_new[idx(i_dir, 0)] = 0.0
        p_new[idx(i_dir, Ntheta-1)] = 0.0

    # check convergence
    diff = p_new - p_vec
    err = np.linalg.norm(diff, np.inf)
    print(f"Iter {it:3d}: ||p_new - p||_inf = {err:.3e}")

    if err < tol:
        p_vec = p_new.copy()
        converged = True
        print(f"Converged in {it} iterations (tol={tol}).")
        break

    # under-relaxation update
    p_vec = relax * p_new + (1.0 - relax) * p_vec

end_time = time.time()
print(f"Elapsed time: {end_time - start_time:.2f} s")

if not converged:
    print("Warning: solver did not converge within max_iter. Consider increasing iterations, decreasing relax or changing tol.")

# reshape for plotting
P = p_vec.reshape((Nr, Ntheta))
P_clip = np.abs(P)

# ------------------------------------------------------------------
# Plot and save results
# ------------------------------------------------------------------
fig = plt.figure(figsize=(12, 5))

# 2D contour in polar coords mapped to Cartesian
ax1 = fig.add_subplot(1, 2, 1)
for i in range(6):
    TH, RR = np.meshgrid(theta + i*np.pi/3, r)
    X = RR * np.cos(TH)
    Y = RR * np.sin(TH)
    cf = ax1.contourf(X, Y, P_clip, vmin=np.min(P_clip), vmax=np.max(P_clip),
                      cmap="viridis", levels=50)
ax1.set_title(rf"Pressure dist. when compressibility $\beta$ = {beta:.1e} Pa")
ax1.set_aspect('equal', 'box')
plt.colorbar(cf, ax=ax1, orientation='vertical')

# 1D vs theta slices
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(theta, P_clip[0, :], '--', label=f"r={r_in:.3f} m", color='C1')
ax2.plot(theta, P_clip[Nr//2, :], '-.', label=f"r={(r_in+r_out)/2:.3f} m", color='C2')
ax2.plot(theta, P_clip[-1, :], ':', label=f"r={r_out:.3f} m", color='C3')
ax2.set_xlabel("theta (rad)")
ax2.set_ylabel("pressure (Pa)")
ax2.legend()
ax2.set_title("Pressure vs theta")

plt.tight_layout()
fig.savefig(out_figure, dpi=200)
print("Figure saved to:", out_figure)

# Print simple diagnostics
print("Peak pressure (abs) [Pa]:", np.max(P_clip))
print("Mean pressure (abs) [Pa]:", np.mean(P_clip))

plt.show()

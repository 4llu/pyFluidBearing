import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Parameters
mu = 0.04             # dynamic viscosity [Pa.s]
omega = 2*np.pi         # angular speed [rad/s]
r_in = 0.05           # inner radius [m]
r_out = 0.10          # outer radius [m]
Delta_theta = np.pi/4 # sector angular width [rad]

hL = 50e-6
hT = 10e-6

# numerical grid
Nr = 40
Ntheta = 40

out_figure = "pressure_contours.png"

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

# Precompute H^3 on grid (here h independent of r)
TH, RR = np.meshgrid(theta, r)
H = h_r_theta(RR, TH)
H3 = H**3

# ------------------------------------------------------------------
# Assemble linear system A p = b using sparse storage
# ------------------------------------------------------------------
N = Nr * Ntheta
rows = []
cols = []
vals = []
b = np.zeros(N)

coef_const = 1.0 / (12.0 * mu)
dh_dtheta = lambda radius: (hT - hL)*radius/(r_out * Delta_theta)

for i in range(Nr):
    ri = r[i]
    for j in range(Ntheta):
        k = idx(i, j)

        # Theta boundaries: Dirichlet p=0
        if j%Ntheta == 0 or (j+1)%Ntheta == 0:
            rows.append(k); cols.append(k); vals.append(1.0)
            b[k] = 0.0
            continue

        # Theta-term coefficients (variable H)
        Hc = H3[i, j]
        Hm = H3[i, j-1]
        Hp = H3[i, j+1]
        A_m = coef_const * Hm / ri
        A_p = coef_const * Hp / ri

        theta_m_coef = -A_m / (dtheta * dtheta)
        theta_p_coef = -A_p / (dtheta * dtheta)
        theta_c_coef = - (theta_m_coef + theta_p_coef)

        # Radial-term coefficients: d/dr( (H^3/(12 mu)) * r * dp/dr )
        if 0 < i < Nr-1:
            H_im = H3[i-1, j]
            H_ip = H3[i+1, j]
            B_im = coef_const * H_im * r[i-1]
            B_ip = coef_const * H_ip * r[i+1]
            rad_m_coef = -B_im / (dr * dr)
            rad_p_coef = -B_ip / (dr * dr)
            rad_c_coef = - (rad_m_coef + rad_p_coef)
        else:
            # Neumann dp/dr = 0: approximate with ghost point p_{-1}=p_{1} at i=0 etc.
            if i == 0:
                H_ip = H3[i+1, j]
                B_ip = coef_const * H_ip * r[i+1]
                rad_m_coef = 0.0
                rad_p_coef = -B_ip / (dr * dr)
                rad_c_coef = -rad_p_coef
            else:  # i == Nr-1
                H_im = H3[i-1, j]
                B_im = coef_const * H_im * r[i-1]
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

        # RHS source term from rotation: (omega * r / 2) * dh/dtheta
        b[k] = (omega * ri / 2.0) * dh_dtheta(ri)

# finalize sparse matrix and solve
A = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
print("Solving linear system (size {})...".format(N))
p_vec = spsolve(A, b)
P = p_vec.reshape((Nr, Ntheta))
P_clip = abs(P)

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
ax1.set_title(f"Pressure dist. when $\omega$ = {np.round(omega,3)} rad/s")
ax1.set_aspect('equal', 'box')
plt.colorbar(cf, ax=ax1, orientation='vertical')

# 1D vs radial slices
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
print("Peak pressure (after clipping) [Pa]:", np.max(P_clip))
print("Mean pressure (after clipping) [Pa]:", np.mean(P_clip))

plt.show()

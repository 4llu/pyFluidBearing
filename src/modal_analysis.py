import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as LA


class Shaft:
    def __init__(self, n1, n2, L, R, E, rho, v):

        self.n1 = n1
        self.n2 = n2
        self.L = L
        self.R = R
        self.E = E
        # self.G = E / (2 * (1 + v))
        self.rho = rho
        # self.v = v
        self.A = np.pi * R**2
        self.I = np.pi * R**4 / 4.0  # second moment area (about centroid)
        self.m = self.rho * self.A  # mass per unit length
        # self.kappa = 6 * (1 + self.v) ** 2 / (7 + 12 * self.v + 4 * self.v**2)
        # self.phi = 12 * E * self.I / (self.kappa * self.G * self.A)

        self.K = self.getK()
        self.M = self.getM()
        self.Gyro = self.getG()
        self.C = np.zeros((8, 8))

    def getM(self):
        m = self.m
        L = self.L
        """ Me = (m / 420.0) * np.array(
            [
                [156, 22 * L, 54, -13 * L],
                [22 * L, 4 * L**2, 13 * L, -3 * L**2],
                [54, 13 * L, 156, -22 * L],
                [-13 * L, -3 * L**2, -22 * L, 4 * L**2],
            ]
        ) """
        Me = (m * L / 420) * np.array(
            [
                [156, 0, 0, 22 * L, 54, 0, 0, -13 * L],
                [0, 156, -22 * L, 0, 0, 54, 13 * L, 0],
                [0, -22 * L, 4 * L**2, 0, 0, -13 * L, -3 * L**2, 0],
                [22 * L, 0, 0, 4 * L**2, 13 * L, 0, 0, -3 * L**2],
                [54, 0, 0, 13 * L, 156, 0, 0, -22 * L],
                [0, 54, -13 * L, 0, 0, 156, 22 * L, 0],
                [0, 13 * L, -3 * L**2, 0, 0, 22 * L, 4 * L**2, 0],
                [-13 * L, 0, 0, -3 * L**2, -22 * L, 0, 0, 4 * L**2],
            ]
        )
        return Me

    def getK(self):
        L = self.L
        k = self.E * self.I / (self.L**3)
        """ Ke = k * np.array(
            [
                [12, 6 * L, -12, 6 * L],
                [6 * L, 4 * L**2, -6 * L, 2 * L**2],
                [-12, -6 * L, 12, -6 * L],
                [6 * L, 2 * L**2, -6 * L, 4 * L**2],
            ]
        ) """
        Ke = k * np.array(
            [
                [12, 0, 0, 6 * L, -12, 0, 0, 6 * L],
                [0, 12, -6 * L, 0, 0, -12, -6 * L, 0],
                [0, -6 * L, 4 * L**2, 0, 0, 6 * L, 2 * L**2, 0],
                [6 * L, 0, 0, 4 * L**2, -6 * L, 0, 0, 2 * L**2],
                [-12, 0, 0, -6 * L, 12, 0, 0, -6 * L],
                [0, -12, 6 * L, 0, 0, 12, 6 * L, 0],
                [0, -6 * L, 2 * L**2, 0, 0, 6 * L, 4 * L**2, 0],
                [6 * L, 0, 0, 2 * L**2, -6 * L, 0, 0, 4 * L**2],
            ]
        )
        return Ke

    def getG(self):
        L = self.L
        g = self.rho * self.I / (15 * L)

        G_shaft = g * np.array(
            [
                [0, 36, -3 * L, 0, 0, -36, -3 * L, 0],
                [-36, 0, 0, -3 * L, 36, 0, 0, -3 * L],
                [3 * L, 0, 0, 4 * L**2, -3 * L, 0, 0, -(L**2)],
                [0, 3 * L, -4 * L**2, 0, 0, -3 * L, L**2, 0],
                [0, -36, 3 * L, 0, 0, 36, 3 * L, 0],
                [36, 0, 0, 3 * L, -36, 0, 0, 3 * L],
                [3 * L, 0, 0, -(L**2), -3 * L, 0, 0, 4 * L**2],
                [0, 3 * L, L**2, 0, 0, -3 * L, -4 * L**2, 0],
            ]
        )

        return G_shaft


# Disk element according Friswell page 158
class Disk:
    def __init__(self, node, R, l, rho, E=None, v=None):
        self.node = node
        self.R = R
        self.l = l
        self.rho = rho
        self.M = self.getM()
        self.K = np.zeros((4, 4))
        self.C = np.zeros((4, 4))
        self.Gyro = self.getG()

    def mass(self):
        return self.rho * np.pi * self.R**2 * self.l

    def I_polar(self):
        m = self.mass()
        return 0.5 * m * self.R**2

    def I_diametral(self):
        m = self.mass()
        return 0.25 * m * self.R**2 + (1.0 / 12.0) * m * self.l**2

    def getM(self):
        m = self.mass()
        I_d = self.I_diametral()
        M_disk = np.array(
            [
                [m, 0, 0, 0],
                [0, m, 0, 0],
                [0, 0, I_d, 0],
                [0, 0, 0, I_d],
            ]
        )
        return M_disk

    def getG(self):

        Ip = self.I_polar()
        G_disk = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, Ip],
                [0, 0, -Ip, 0],
            ]
        )
        return G_disk


class Bearing:
    def __init__(self, node, kxx=0.0, kyy=0.0, cxx=0.0, cyy=0.0):
        self.node = node
        self.kxx = kxx
        self.kyy = kyy
        self.cxx = cxx
        self.cyy = cyy
        self.M = np.zeros((4, 4))
        self.K = np.array([[kxx, 0, 0, 0], [0, kyy, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.C = np.array([[cxx, 0, 0, 0], [0, cyy, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.Gyro = np.zeros((4, 4))


class RotorSystem:
    def __init__(self, bearings):
        # Material and geometry
        rho = 7810.0
        E = 211e9
        v = 0.3

        r_disk = 0.25 / 2.0
        l_disk = 0.04
        r_shaft = 0.025 / 2.0
        l_shaft = 0.25

        n_nodes = 7

        shafts = []
        for i in range(n_nodes - 1):
            el = Shaft(n1=i, n2=i + 1, L=l_shaft, R=r_shaft, E=E, rho=rho, v=v)
            shafts.append(el)

        # Disk at node 6
        disk_1 = Disk(node=6, R=r_disk, l=l_disk, rho=rho)
        disks = [disk_1]
        self.n_nodes = n_nodes
        self.ndof_per_node = 4
        self.Ndof = n_nodes * self.ndof_per_node
        self.shafts = shafts
        self.disks = disks
        self.bearings = bearings
        self.M = np.zeros((self.Ndof, self.Ndof), dtype=float)
        self.K = np.zeros((self.Ndof, self.Ndof), dtype=float)
        self.C = np.zeros((self.Ndof, self.Ndof), dtype=float)
        self.G = np.zeros((self.Ndof, self.Ndof), dtype=float)
        self.assemble()

    def assemble(self):
        # assemble shafts
        for i, s in enumerate(self.shafts):
            base = s.n1 * self.ndof_per_node
            for i in range(8):
                for j in range(8):
                    self.M[base + i, base + j] += s.M[i, j]
                    self.G[base + i, base + j] += s.Gyro[i, j]
                    self.K[base + i, base + j] += s.K[i, j]
                    self.C[base + i, base + j] += s.C[i, j]

        # assemble disks
        for d in self.disks:
            base = d.node * self.ndof_per_node
            for i in range(4):
                for j in range(4):
                    self.M[base + i, base + j] += d.M[i, j]
                    self.G[base + i, base + j] += d.Gyro[i, j]
                    self.K[base + i, base + j] += d.K[i, j]
                    self.C[base + i, base + j] += d.C[i, j]
        # assemble bearings
        for b in self.bearings:
            base = b.node * self.ndof_per_node
            for i in range(4):
                for j in range(4):
                    self.K[base + i, base + j] += b.K[i, j]
                    self.C[base + i, base + j] += b.C[i, j]

    def state_space_matrix(self, Omega=0.0):
        N = self.Ndof
        Z = np.zeros((N, N))
        I = np.eye(N)
        Minv = LA.inv(self.M)
        A_top = np.concatenate((Z, I), axis=1)
        K_mat = self.K
        A_bottom = np.concatenate(
            (-Minv @ K_mat, -Minv @ (self.C + Omega * self.G)), axis=1
        )
        A = np.concatenate((A_top, A_bottom), axis=0)
        return A

    def eigenvalues(self, Omega=0.0):
        A = self.state_space_matrix(Omega=Omega)
        eigvals, eigvecs = LA.eig(A)
        eigvals = sorted(eigvals, key=np.abs)
        return eigvals, eigvecs

    def campbell(self, speeds, out_figure="reports/campbell.png", savefig=False):
        eig_1 = np.zeros_like(speeds)
        eig_2 = np.zeros_like(speeds)
        eig_3 = np.zeros_like(speeds)
        eig_4 = np.zeros_like(speeds)
        eig_5 = np.zeros_like(speeds)
        eig_6 = np.zeros_like(speeds)

        for i, speed in enumerate(speeds):
            eig, _ = self.eigenvalues(Omega=speed / 60 * 2 * np.pi)
            eig_1[i] = np.abs(eig[0])
            eig_2[i] = np.abs(eig[2])
            eig_3[i] = np.abs(eig[4])
            eig_4[i] = np.abs(eig[6])
            eig_5[i] = np.abs(eig[8])
            eig_6[i] = np.abs(eig[10])

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        ax.plot(speeds, speeds / 60, "b--", label="Rotor speed")
        ax.plot(speeds, eig_1 / (2 * np.pi), label="First")
        ax.plot(speeds, eig_2 / (2 * np.pi), label="Second")
        ax.plot(speeds, eig_3 / (2 * np.pi), label="Third")
        ax.plot(speeds, eig_4 / (2 * np.pi), label="Fourth")
        ax.plot(speeds, eig_5 / (2 * np.pi), label="Fifth")
        ax.plot(speeds, eig_6 / (2 * np.pi), label="Sixth")
        ax.tick_params(labelsize=11)

        fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1), fontsize=11)
        ax.set_xlabel("Speed (RPM)", fontsize=11)
        ax.set_ylabel("Natural frequency (Hz)", fontsize=11)
        plt.tight_layout()
        plt.subplots_adjust(top=0.83)

        if savefig:
            fig.savefig(os.path.join("reports", out_figure), dpi=200)
            print(f"Figure saved to: {os.path.join('reports', out_figure)}")

        plt.show()

    def print_eigs(self):

        eig_0, _ = self.eigenvalues(Omega=0.0)
        print("Eigenvalues at Omega = 0:")
        for i in range(8):
            print(f"{np.real(eig_0[i]):.2f} {np.imag(eig_0[i]):.2f}")

        rpm = 1000.0
        Omega = rpm / 60.0 * 2.0 * np.pi
        eig_1000, _ = self.eigenvalues(Omega=Omega)
        print("\nEigenvalues at Omega = 1000 rpm:")
        for i in range(8):
            print(f"{np.real(eig_1000[i]):.2f} {np.imag(eig_1000[i]):.2f}")

        rpm = 3000.0
        Omega = rpm / 60.0 * 2.0 * np.pi
        eig_3000, _ = self.eigenvalues(Omega=Omega)
        print("\nEigenvalues at Omega = 3000 rpm:")
        for i in range(8):
            print(f"{np.real(eig_3000[i]):.2f} {np.imag(eig_3000[i]):.2f}")


if __name__ == "__main__":
    # Bearings at node 0 and 4
    bearing_1 = Bearing(node=0, kxx=0.2e6, kyy=0.4e6, cxx=0.0, cyy=0.0)
    bearing_2 = Bearing(node=4, kxx=0.2e6, kyy=0.4e6, cxx=0.0, cyy=0.0)
    bearings = [bearing_1, bearing_2]

    rotor = RotorSystem(bearings=bearings)
    speeds = np.linspace(0, 3000, 1000)
    rotor.campbell(speeds)
    rotor.print_eigs()

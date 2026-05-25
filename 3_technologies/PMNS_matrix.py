import numpy as np

# NuFIT 5.2 best-fit, normal ordering
SIN2_TH12 = 0.304
SIN2_TH23 = 0.573
SIN2_TH13 = 0.02219
DELTA_CP_DEG = 197  # CP phase in degrees
delta_cp = np.deg2rad(DELTA_CP_DEG)

th12 = np.arcsin(np.sqrt(SIN2_TH12))
th23 = np.arcsin(np.sqrt(SIN2_TH23))
th13 = np.arcsin(np.sqrt(SIN2_TH13))
c12, s12 = np.cos(th12), np.sin(th12)
c23, s23 = np.cos(th23), np.sin(th23)
c13, s13 = np.cos(th13), np.sin(th13)

U = np.array([
    [c12*c13,
     s12*c13,
     s13*np.exp(-1j * delta_cp)],

    [-s12*c23 - c12*s23*s13*np.exp(1j * delta_cp),
     c12*c23 - s12*s23*s13*np.exp(1j * delta_cp),
     s23*c13],

    [s12*s23 - c12*c23*s13*np.exp(1j * delta_cp),
     -c12*s23 - s12*c23*s13*np.exp(1j * delta_cp),
     c23*c13]
], dtype=complex)

# |U_{αi}|²
Usq = np.abs(U)**2

# P[α, β] = Σ_i |U_{αi}|² |U_{βi}|²
P = Usq @ Usq.T
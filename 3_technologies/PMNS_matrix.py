import numpy as np

# NuFIT 5.2 best-fit, normal ordering; delta_CP = 0 (averaged formula, decoherence limit)
SIN2_TH12 = 0.307
SIN2_TH23 = 0.546
SIN2_TH13 = 0.02220

th12 = np.arcsin(np.sqrt(SIN2_TH12))
th23 = np.arcsin(np.sqrt(SIN2_TH23))
th13 = np.arcsin(np.sqrt(SIN2_TH13))
c12, s12 = np.cos(th12), np.sin(th12)
c23, s23 = np.cos(th23), np.sin(th23)
c13, s13 = np.cos(th13), np.sin(th13)

# |U_{αi}|²  rows: flavour α = (e, μ, τ),  cols: mass eigenstate i = (1, 2, 3)
Usq = np.array([
    [c12**2 * c13**2,  s12**2 * c13**2,  s13**2],
    [s12**2*c23**2 + c12**2*s23**2*s13**2,
     c12**2*c23**2 + s12**2*s23**2*s13**2,
     s23**2 * c13**2],
    [s12**2*s23**2 + c12**2*c23**2*s13**2,
     c12**2*s23**2 + s12**2*c23**2*s13**2,
     c23**2 * c13**2],
])

# P[α, β] = P(ν_α → ν_β) = Σ_i |U_{αi}|²|U_{βi}|²
P = Usq @ Usq.T

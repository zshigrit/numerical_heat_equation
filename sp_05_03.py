"""
Supplemental program 5.3

Use implicit formulation with "excess heat" or "apparent heat capacity" to solve for soil temperatures with phase change in comparison with
Neumann's analytical solution.

Translated from the original MATLAB code.
"""
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Physcon:
    tfrz: float = 273.15
    rhowat: float = 1000.0
    rhoice: float = 917.0
    hfus: float = 0.3337e6

@dataclass
class SoilVar:
    method: str = 'apparent-heat-capacity'
    nsoi: int = 60
    dz: np.ndarray = field(default_factory=lambda: np.full(60, 0.10))
    z_plus_onehalf: np.ndarray = field(init=False)
    z: np.ndarray = field(init=False)
    dz_plus_onehalf: np.ndarray = field(init=False)
    tsoi: np.ndarray = field(init=False)
    h2osoi_liq: np.ndarray = field(init=False)
    h2osoi_ice: np.ndarray = field(init=False)
    tk: np.ndarray = field(init=False)
    cv: np.ndarray = field(init=False)
    gsoi: float = 0.0
    hfsoi: float = 0.0

    def __post_init__(self):
        self.z_plus_onehalf = np.zeros(self.nsoi)
        self.z_plus_onehalf[0] = -self.dz[0]
        for i in range(1, self.nsoi):
            self.z_plus_onehalf[i] = self.z_plus_onehalf[i-1] - self.dz[i]

        self.z = np.zeros(self.nsoi)
        self.z[0] = 0.5 * self.z_plus_onehalf[0]
        for i in range(1, self.nsoi):
            self.z[i] = 0.5 * (self.z_plus_onehalf[i-1] + self.z_plus_onehalf[i])

        self.dz_plus_onehalf = np.zeros(self.nsoi)
        for i in range(self.nsoi-1):
            self.dz_plus_onehalf[i] = self.z[i] - self.z[i+1]
        self.dz_plus_onehalf[-1] = 0.5 * self.dz[-1]

        self.tsoi = np.full(self.nsoi, Physcon().tfrz + 2.0)
        self.h2osoi_liq = np.zeros(self.nsoi)
        self.h2osoi_ice = np.zeros(self.nsoi)
        for i in range(self.nsoi):
            if self.tsoi[i] > Physcon().tfrz:
                self.h2osoi_liq[i] = 0.187 * 1770 * self.dz[i]
            else:
                self.h2osoi_ice[i] = 0.187 * 1770 * self.dz[i]

        self.tk = np.zeros(self.nsoi)
        self.cv = np.zeros(self.nsoi)


def tridiagonal_solver(a, b, c, d):
    """Solve the tridiagonal system R * u = d as in the original MATLAB
    tridiagonal_solver.m routine."""
    n = len(d)
    e = np.zeros(n)
    f = np.zeros(n)
    e[0] = c[0] / b[0]
    for i in range(1, n-1):
        e[i] = c[i] / (b[i] - a[i] * e[i-1])
    f[0] = d[0] / b[0]
    for i in range(1, n):
        f[i] = (d[i] - a[i] * f[i-1]) / (b[i] - a[i] * e[i-1])
    u = np.zeros(n)
    u[-1] = f[-1]
    for i in range(n-2, -1, -1):
        u[i] = f[i] - e[i] * u[i+1]
    return u


def soil_thermal_properties(physcon: Physcon, soilvar: SoilVar) -> SoilVar:
    """Calculate soil thermal conductivity and heat capacity.
    This follows the notes in soil_thermal_properties.m.
    """
    tinc = 0.5
    tku = 1.860
    tkf = 2.324
    cvu = 2.862e6
    cvf = 1.966e6

    for i in range(soilvar.nsoi):
        watliq = soilvar.h2osoi_liq[i] / (physcon.rhowat * soilvar.dz[i])
        watice = soilvar.h2osoi_ice[i] / (physcon.rhoice * soilvar.dz[i])
        ql = physcon.hfus * (physcon.rhowat * watliq + physcon.rhoice * watice)
        if soilvar.tsoi[i] > physcon.tfrz + tinc:
            soilvar.cv[i] = cvu
            soilvar.tk[i] = tku
        elif soilvar.tsoi[i] >= physcon.tfrz - tinc and soilvar.tsoi[i] <= physcon.tfrz + tinc:
            if soilvar.method == 'apparent-heat-capacity':
                soilvar.cv[i] = (cvf + cvu) / 2 + ql / (2 * tinc)
            else:  # excess-heat
                soilvar.cv[i] = (cvf + cvu) / 2
            soilvar.tk[i] = tkf + (tku - tkf) * (soilvar.tsoi[i] - physcon.tfrz + tinc) / (2 * tinc)
        else:
            soilvar.cv[i] = cvf
            soilvar.tk[i] = tkf
    return soilvar


def phase_change(physcon: Physcon, soilvar: SoilVar, dt: float) -> SoilVar:
    """Adjust soil temperature for phase change.
    Based on phase_change.m where excess or deficit energy is used to freeze or melt ice."""
    soilvar.hfsoi = 0.0
    for i in range(soilvar.nsoi):
        wliq0 = soilvar.h2osoi_liq[i]
        wice0 = soilvar.h2osoi_ice[i]
        wmass0 = wliq0 + wice0
        tsoi0 = soilvar.tsoi[i]
        imelt = 0
        if soilvar.h2osoi_ice[i] > 0 and soilvar.tsoi[i] > physcon.tfrz:
            imelt = 1
            soilvar.tsoi[i] = physcon.tfrz
        if soilvar.h2osoi_liq[i] > 0 and soilvar.tsoi[i] < physcon.tfrz:
            imelt = 2
            soilvar.tsoi[i] = physcon.tfrz
        if imelt > 0:
            heat_flux_pot = (soilvar.tsoi[i] - tsoi0) * soilvar.cv[i] * soilvar.dz[i] / dt
        else:
            heat_flux_pot = 0.0
        if imelt == 1:
            heat_flux_max = -soilvar.h2osoi_ice[i] * physcon.hfus / dt
        elif imelt == 2:
            heat_flux_max = soilvar.h2osoi_liq[i] * physcon.hfus / dt
        if imelt > 0:
            ice_flux = heat_flux_pot / physcon.hfus
            soilvar.h2osoi_ice[i] = wice0 + ice_flux * dt
            soilvar.h2osoi_ice[i] = max(0.0, soilvar.h2osoi_ice[i])
            soilvar.h2osoi_ice[i] = min(wmass0, soilvar.h2osoi_ice[i])
            soilvar.h2osoi_liq[i] = max(0.0, wmass0 - soilvar.h2osoi_ice[i])
            heat_flux = physcon.hfus * (soilvar.h2osoi_ice[i] - wice0) / dt
            soilvar.hfsoi += heat_flux
            residual = heat_flux_pot - heat_flux
            soilvar.tsoi[i] = soilvar.tsoi[i] - residual * dt / (soilvar.cv[i] * soilvar.dz[i])
    return soilvar


def soil_temperature(physcon: Physcon, soilvar: SoilVar, tsurf: float, dt: float) -> SoilVar:
    """Use an implicit formulation with the surface temperature boundary to
    solve for soil temperatures at the next time step. Based on
    soil_temperature.m from the MATLAB code."""
    tsoi0 = soilvar.tsoi.copy()
    tk_plus_onehalf = np.zeros(soilvar.nsoi - 1)
    for i in range(soilvar.nsoi - 1):
        tk_plus_onehalf[i] = (
            soilvar.tk[i] * soilvar.tk[i + 1] * (soilvar.z[i] - soilvar.z[i + 1]) /
            (soilvar.tk[i] * (soilvar.z_plus_onehalf[i] - soilvar.z[i + 1]) + soilvar.tk[i + 1] * (soilvar.z[i] - soilvar.z_plus_onehalf[i]))
        )
    a = np.zeros(soilvar.nsoi)
    b = np.zeros(soilvar.nsoi)
    c = np.zeros(soilvar.nsoi)
    d = np.zeros(soilvar.nsoi)
    i = 0
    m = soilvar.cv[i] * soilvar.dz[i] / dt
    a[i] = 0.0
    c[i] = -tk_plus_onehalf[i] / soilvar.dz_plus_onehalf[i]
    b[i] = m - c[i] + soilvar.tk[i] / (0 - soilvar.z[i])
    d[i] = m * soilvar.tsoi[i] + soilvar.tk[i] / (0 - soilvar.z[i]) * tsurf
    for i in range(1, soilvar.nsoi - 1):
        m = soilvar.cv[i] * soilvar.dz[i] / dt
        a[i] = -tk_plus_onehalf[i - 1] / soilvar.dz_plus_onehalf[i - 1]
        c[i] = -tk_plus_onehalf[i] / soilvar.dz_plus_onehalf[i]
        b[i] = m - a[i] - c[i]
        d[i] = m * soilvar.tsoi[i]
    i = soilvar.nsoi - 1
    m = soilvar.cv[i] * soilvar.dz[i] / dt
    a[i] = -tk_plus_onehalf[i - 1] / soilvar.dz_plus_onehalf[i - 1]
    c[i] = 0.0
    b[i] = m - a[i]
    d[i] = m * soilvar.tsoi[i]
    soilvar.tsoi = tridiagonal_solver(a, b, c, d)
    soilvar.gsoi = soilvar.tk[0] * (tsurf - soilvar.tsoi[0]) / (0 - soilvar.z[0])
    if soilvar.method == 'apparent-heat-capacity':
        soilvar.hfsoi = 0.0
    else:
        soilvar = phase_change(physcon, soilvar, dt)
    edif = np.sum(soilvar.cv * soilvar.dz * (soilvar.tsoi - tsoi0) / dt)
    err = edif - soilvar.gsoi - soilvar.hfsoi
    if abs(err) > 1e-3:
        raise RuntimeError('Soil temperature energy conservation error')
    return soilvar


def neumann():
    """Analytical solution for the Neumann problem from Lunardini (1981)."""
    nsoi = 4
    depth = np.array([0.25, 0.55, 0.85, 1.15])
    ndays = 60
    ts = -10.0
    t0 = np.full(nsoi, 2.0)
    tf = 0.0
    a1 = 42.55 / 3600.0 / (100 * 100)
    a2 = 23.39 / 3600.0 / (100 * 100)
    rm = 3.6 / (60 * 100)
    rg = rm / (2 * np.sqrt(a1))
    m = 0
    iday_out = [0]
    xf_out = [0.0]
    t1_out = [t0[0]]
    t2_out = [t0[1]]
    t3_out = [t0[2]]
    t4_out = [t0[3]]
    z1_out = [depth[0] * 100]
    z2_out = [depth[1] * 100]
    z3_out = [depth[2] * 100]
    z4_out = [depth[3] * 100]
    for iday in range(1, ndays + 1):
        time = iday * 24 * 3600
        xf = 2 * rg * np.sqrt(a1 * time)
        t = np.zeros(nsoi)
        for i in range(nsoi):
            if depth[i] <= xf:
                x = depth[i] / (2 * np.sqrt(a1 * time))
                y = rg
                t[i] = ts + (tf - ts) * (np.math.erf(x) / np.math.erf(y))
            else:
                x = depth[i] / (2 * np.sqrt(a2 * time))
                y = rg * np.sqrt(a1 / a2)
                t[i] = t0[i] - (t0[i] - tf) * (1 - np.math.erf(x)) / (1 - np.math.erf(y))
        iday_out.append(iday)
        xf_out.append(xf * 100)
        t1_out.append(t[0])
        t2_out.append(t[1])
        t3_out.append(t[2])
        t4_out.append(t[3])
        z1_out.append(depth[0] * 100)
        z2_out.append(depth[1] * 100)
        z3_out.append(depth[2] * 100)
        z4_out.append(depth[3] * 100)
    A = np.vstack([iday_out, xf_out, z1_out, t1_out, z2_out, t2_out, z3_out, t3_out, z4_out, t4_out]).T
    np.savetxt('data_analytical.txt', A, fmt='%10.3f', header=' day       z0C        z1         t1         z2         t2         z3         t3         z4         t4')


def main():
    """Run the numerical solution and compare with the Neumann analytical solution."""
    physcon = Physcon()
    dt = 3600.0
    nday = 60
    soilvar = SoilVar(method='apparent-heat-capacity')
    k1, k2, k3, k4 = 3, 6, 9, 12
    iday_out = [0]
    d0c_out = [0.0]
    z1_out = [-soilvar.z[k1-1] * 100]
    tsoi1_out = [soilvar.tsoi[k1-1] - physcon.tfrz]
    z2_out = [-soilvar.z[k2-1] * 100]
    tsoi2_out = [soilvar.tsoi[k2-1] - physcon.tfrz]
    z3_out = [-soilvar.z[k3-1] * 100]
    tsoi3_out = [soilvar.tsoi[k3-1] - physcon.tfrz]
    z4_out = [-soilvar.z[k4-1] * 100]
    tsoi4_out = [soilvar.tsoi[k4-1] - physcon.tfrz]

    ntim = round(86400 / dt)
    for iday in range(1, nday + 1):
        for itim in range(ntim):
            tsurf = -10 + physcon.tfrz
            soilvar = soil_thermal_properties(physcon, soilvar)
            soilvar = soil_temperature(physcon, soilvar, tsurf, dt)
            d0c = 0.0
            if soilvar.method == 'excess-heat':
                num_z = 0
                sum_z = 0.0
                for i in range(soilvar.nsoi):
                    if abs(soilvar.tsoi[i] - physcon.tfrz) < 1e-3:
                        num_z += 1
                        sum_z += soilvar.z[i]
                if num_z > 0:
                    d0c = sum_z / num_z
            else:
                for i in range(1, soilvar.nsoi):
                    if soilvar.tsoi[i-1] <= physcon.tfrz and soilvar.tsoi[i] > physcon.tfrz:
                        b = (soilvar.tsoi[i] - soilvar.tsoi[i-1]) / (soilvar.z[i] - soilvar.z[i-1])
                        a = soilvar.tsoi[i] - b * soilvar.z[i]
                        d0c = (physcon.tfrz - a) / b
                        break
            if itim == ntim - 1:
                iday_out.append(iday)
                d0c_out.append(-d0c * 100)
                z1_out.append(-soilvar.z[k1-1] * 100)
                tsoi1_out.append(soilvar.tsoi[k1-1] - physcon.tfrz)
                z2_out.append(-soilvar.z[k2-1] * 100)
                tsoi2_out.append(soilvar.tsoi[k2-1] - physcon.tfrz)
                z3_out.append(-soilvar.z[k3-1] * 100)
                tsoi3_out.append(soilvar.tsoi[k3-1] - physcon.tfrz)
                z4_out.append(-soilvar.z[k4-1] * 100)
                tsoi4_out.append(soilvar.tsoi[k4-1] - physcon.tfrz)
    A = np.vstack([iday_out, d0c_out, z1_out, tsoi1_out, z2_out, tsoi2_out, z3_out, tsoi3_out, z4_out, tsoi4_out]).T
    np.savetxt('data_numerical.txt', A, fmt='%10.3f', header=' day       z0C        z1         t1         z2         t2         z3         t3         z4         t4')
    neumann()


if __name__ == '__main__':
    main()

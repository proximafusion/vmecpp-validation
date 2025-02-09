# SPDX-FileCopyrightText: 2025-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
# The default tolerance for value comparisons
_DEFAULT_TOL = 1e-16

# A dictionary of tolerances for the various V&V checks.
# The key is the (case-sensitive) variable name (e.g. 'rmnc', 'DShear', ...) to which
# the tolerance applies.
# If no value is found for a given variable, _DEFAULT_TOL is used instead.
# If the value is a tuple, it's (fixed-boundary tolerance, free-boundary tolerance),
# otherwise the same is used for both cases.
_TOLERANCES: dict[str, float | tuple[float, float]] = {
    # As per the V&V spec, we only require the number of iterations to
    # convergence are in a 20% range from the reference.
    "niter": 0.2,
    # For these quantities we only care about matching the right order of
    # magnitude (as per the V&V spec), so the default tolerance is 1.
    "fsqr": 1.0,
    "fsqz": 1.0,
    "fsql": 1.0,
    # Scalar quantities
    "wb": (1.0e-12, 1.0e-9),  # w7x_beta1_mn12_ns25 is the "worst offender"
    "wp": (1.0e-12, 1.0e-8),  # w7x_beta1_mn12_ns51
    "volavgB": (1.0e-10, 1.0e-6),  # cth_like_beta0_mn12_ns25
    "rbtor": (1.0e-10, 2.0e-8),
    "rbtor0": (5.0e-10, 1.0e-8),
    "ctor": 1.0e-8,
    "betatotal": (1.0e-9, 1.0e-7),  # cth_like_beta1_mn12_ns99
    "betapol": (1.0e-9, 5.0e-7),
    "betator": (1.0e-9, 1.0e-7),  # cth_like_beta1_mn12_ns99
    "betaxis": (1.0e-8, 5.0e-8),
    "b0": (1.0e-8, 5.0e-7),  # cth_like_beta1_mn12_ns99
    "IonLarmor": (1.0e-12, 5.0e-8),  # cth_like_beta1_mn12_ns99
    "aspect": (1.0e-13, 1.0e-6),  # cth_like_beta0_mn12_ns25
    "Aminor_p": (1.0e-13, 2.0e-7),  # cth_like_beta0_mn12_ns25
    "Rmajor_p": (1.0e-13, 1.0e-6),  # cth_like_beta0_mn12_ns25
    "volume_p": (1.0e-13, 5.0e-6),  # cth_like_beta0_mn12_ns51
    # Radial profiles
    "pres": 1.0e-11,
    "presf": (1.0e-11, 5.0e-11),
    "mass": 1.0e-11,
    "iotaf": (1.0e-7, 1.0e-4),  # cth_like_beta0_mn12_ns51
    "q_factor": (1.0e-6, 1.0e-4),  # cth_like_beta0_mn12_ns51
    "chi": (1.0e-8, 1.0e-5),  # w7x_beta1_mn12_ns25
    "chipf": (1.0e-8, 5.0e-5),  # cth_like_beta0_mn12_ns51
    "iotas": (1.0e-7, 5.0e-5),  # cth_like_beta0_mn12_ns51
    "buco": (1.0e-11, 1.0e-9),
    "bvco": (1.0e-9, 5.0e-8),
    "vp": (1.0e-9, 1.0e-6),  # cth_like_beta0_mn12_ns51
    "specw": (1.0e-5, 1.0e-2),  # cth_like_beta0_mn12_ns25
    "over_r": (1.0e-7, 2.0e-5),  # cth_like_beta0_mn12_ns99
    "bdotgradv": (1.0e-8, 5.0e-5),  # cth_like_beta0_mn12_ns99
    "beta_vol": (1.0e-8, 5.0e-7),
    "jdotb": (0.2, 0.5),  # we expect bad matching between VMEC++ and Fortran here
    "jcuru": (1.0e-3, 0.5),  # cth_like_beta0_mn12_ns99
    "jcurv": (1.0e-5, 1.0e-3),
    "equif": (1.0e-4, 1.1),  # cth_like_beta0_mn12_ns51
    "DMerc": (2.0e-3, 2.0e-2),  # cth_like_beta0_mn12_ns99
    "DShear": (1.0e-5, 5.0e-3),  # cth_like_beta0_mn12_ns25
    "DWell": (1.0e-4, 1.0e-2),
    "DCurr": (1.0e-4, 5.0e-2),  # cth_like_beta0_mn12_ns51
    "DGeod": (1.0e-5, 5.0e-2),  # cth_like_beta0_mn12_ns51
    "raxis_cc": (1.0e-8, 2.0e-6),
    "zaxis_cs": (1.0e-8, 5.0e-6),
    # Fourier coefficients per surface
    "rmnc": (1.0e-7, 2.0e-5),
    "zmns": (1.0e-7, 2.0e-5),
    "lmns": (1.0e-5, 1.0e-3),  # cth_like_beta0_mn12_ns51
    "bmnc": (1.0e-7, 2.0e-5),
    "gmnc": (1.0e-7, 2.0e-4),  # w7x_beta1_mn12_ns51
    "bsubumnc": (1.0e-7, 5.0e-5),
    "bsubvmnc": (1.0e-7, 1.0e-4),
    "bsubsmns": (1.0e-6, 1.0e-4),  # cth_like_beta0_mn12_ns25
    "bsupumnc": (1.0e-6, 2.0e-3),  # cth_like_beta0_mn12_ns25
    "bsupvmnc": (1.0e-7, 5.0e-5),  # cth_like_beta0_mn12_ns51
    # cth_like_beta0_mn12_ns51
    "vmec_max_normalized_geodesic_curvature": (1.0e-5, 5.0e-3),
    # cth_like_beta0_mn12_ns51
    "vmec_mean_normalized_geodesic_curvature": (1.0e-6, 5.0e-4),
    "vmec_min_normalized_magnetic_gradient_scale_length": (1.0e-6, 5.0e-3),
    # cth_like_beta0_mn12_ns51
    "vmec_mean_normalized_magnetic_gradient_scale_length": (1.0e-6, 5.0e-3),
    "proxima_qi.bounce_point_residuals_profile": (1.0e-7, 2.0e-4),
    "proxima_qi.J_normalized_residuals_profile": (1.0e-8, 1.0e-4),
    # We observe numerical noise in the NEO calculations for some radial regions of
    # some equilibria, which is why we allow for rather large deviations here
    "epsilon_effective": (1.0e-3, 5.0e-3),
    "txport.g11": (1.0e-5, 5.0e-4),
    "qi.profile": (1.0e-9, 2.0e-6),
    "bmnc_b": (1.0e-7, 5.0e-5),
    # realspace geometry and magnetic field components
    "flux_surface_geometry_r": (1.0e-7, 5.0e-5),  # cth_like_beta0_mn12_ns51
    "flux_surface_geometry_z": (1.0e-7, 5.0e-5),  # cth_like_beta0_mn12_ns51
    "flux_surface_b_r": (1.0e-7, 5.0e-5),  # cth_like_beta0_mn12_ns51
    "flux_surface_b_phi": (1.0e-7, 5.0e-5),  # cth_like_beta0_mn12_ns51
    "flux_surface_b_z": (1.0e-6, 1.0e-4),
    # free-boundary quantities
    "rmax_surf": (1.0e-12, 5.0e-5),  # cth_like_beta0_mn12_ns25
    "rmin_surf": (1.0e-12, 1.0e-5),  # cth_like_beta0_mn12_ns25
    "zmax_surf": (1.0e-12, 1.0e-4),  # cth_like_beta0_mn12_ns51
}


# So client code does not directly handle (and accidentally modify) _TOLERANCES.
def get_tolerance(varname: str, case_name: str) -> float:
    tolerance = _TOLERANCES.get(varname, _DEFAULT_TOL)

    if isinstance(tolerance, tuple):
        # must pick the right one between fixed- and free-boundary
        if case_name in ["w7x", "ncsx", "cth_like"]:
            return tolerance[1]
        else:
            assert case_name in [
                "cm_a",
                "cm_b",
                "dshape",
                "heliotron",
                "precise_qa",
                "precise_qh",
            ]
            return tolerance[0]
    else:
        return tolerance

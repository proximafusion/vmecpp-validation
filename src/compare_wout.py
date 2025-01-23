#!/usr/bin/env python
#
# Utilities to compare contents of two wout files. Can also be used as a script:
# python compare_wout.py wout_CONFIG.vmecpp.nc wout_CONFIG.nc

import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numba
import numpy as np
import rich.box
import rich.console
from jaxtyping import Float
from src._plotting import plot_diffs
from src._utils import (
    ArrayCheck,
    ScalarCheck,
    Status,
    VnvChecks,
    case_name_from_conf_name,
    check_contents,
    conf_name_from_path,
    log_error,
)
from src.tolerances import get_tolerance

logger = logging.getLogger("validate_vmec.compare_wout")
logger.addHandler(logging.NullHandler())
console = rich.console.Console()


def check_shapes(test_var, ref_var) -> Status:
    if test_var.dimensions != ref_var.dimensions:
        msg = f"Variable {test_var.name} has wrong dimensions:"
        msg += f"\n\tunder test: {test_var.dimensions}"
        msg += f"\n\treference : {ref_var.dimensions}"
        log_error(msg, logger)
        return Status.MISMATCH

    if test_var.shape != ref_var.shape:
        msg = f"Variable {test_var.name} has wrong shape:"
        msg += f"\n\tunder test: {test_var.shape}"
        msg += f"\n\treference : {ref_var.shape}"
        log_error(msg, logger)
        return Status.MISMATCH

    return Status.OK


@numba.njit
def flux_surface_geom_rz(
    surface_idx: int,
    theta_grid: Float[np.ndarray, " num_theta"],
    phi_grid: Float[np.ndarray, " num_phi"],
    xm: Float[np.ndarray, " xmsize"],
    xn: Float[np.ndarray, " xnsize"],
    test_rmnc: Float[np.ndarray, "ns mn"],
    test_zmns: Float[np.ndarray, "ns mn"],
    ref_rmnc: Float[np.ndarray, "ns mn"],
    ref_zmns: Float[np.ndarray, "ns mn"],
) -> tuple[
    Float[np.ndarray, "num_theta num_phi"],
    Float[np.ndarray, "num_theta num_phi"],
    Float[np.ndarray, "num_theta num_phi"],
    Float[np.ndarray, "num_theta num_phi"],
]:
    num_theta: int = theta_grid.size
    num_phi: int = phi_grid.size

    test_r = np.zeros((num_theta, num_phi))
    test_z = np.zeros((num_theta, num_phi))

    ref_r = np.zeros((num_theta, num_phi))
    ref_z = np.zeros((num_theta, num_phi))

    j = surface_idx
    for l, theta in enumerate(theta_grid):  # noqa: E741
        for k, phi in enumerate(phi_grid):
            kernel = xm * theta - xn * phi

            cos_kernel = np.cos(kernel)
            sin_kernel = np.sin(kernel)

            test_r[l, k] = np.sum(test_rmnc[j, :] * cos_kernel)
            test_z[l, k] = np.sum(test_zmns[j, :] * sin_kernel)

            ref_r[l, k] = np.sum(ref_rmnc[j, :] * cos_kernel)
            ref_z[l, k] = np.sum(ref_zmns[j, :] * sin_kernel)

    return test_r, test_z, ref_r, ref_z


@numba.njit
def flux_surface_geom_brbpbz(
    surface_idx: int,
    theta_grid: Float[np.ndarray, " num_theta"],
    phi_grid: Float[np.ndarray, " num_phi"],
    xm: Float[np.ndarray, " xmsize"],
    xn: Float[np.ndarray, " xnsize"],
    xm_nyq: Float[np.ndarray, " xmnyqsize"],
    xn_nyq: Float[np.ndarray, " xnnyqsize"],
    test_rmnc: Float[np.ndarray, "ns mn"],
    test_zmns: Float[np.ndarray, "ns mn"],
    ref_rmnc: Float[np.ndarray, "ns mn"],
    ref_zmns: Float[np.ndarray, "ns mn"],
    test_r: Float[np.ndarray, "num_theta num_phi"],
    ref_r: Float[np.ndarray, "num_theta num_phi"],
    test_bsupumnc: Float[np.ndarray, "..."],
    test_bsupvmnc: Float[np.ndarray, "..."],
    ref_bsupumnc: Float[np.ndarray, "..."],
    ref_bsupvmnc: Float[np.ndarray, "..."],
) -> tuple[
    Float[np.ndarray, "num_theta num_phi"],
    Float[np.ndarray, "num_theta num_phi"],
    Float[np.ndarray, "num_theta num_phi"],
    Float[np.ndarray, "num_theta num_phi"],
    Float[np.ndarray, "num_theta num_phi"],
    Float[np.ndarray, "num_theta num_phi"],
]:
    num_theta: int = theta_grid.size
    num_phi: int = phi_grid.size

    test_br = np.zeros((num_theta, num_phi))
    test_bp = np.zeros((num_theta, num_phi))
    test_bz = np.zeros((num_theta, num_phi))

    ref_br = np.zeros((num_theta, num_phi))
    ref_bp = np.zeros((num_theta, num_phi))
    ref_bz = np.zeros((num_theta, num_phi))

    j = surface_idx
    for l, theta in enumerate(theta_grid):  # noqa: E741
        for k, phi in enumerate(phi_grid):
            kernel = xm * theta - xn * phi

            cos_kernel = np.cos(kernel)
            sin_kernel = np.sin(kernel)

            # dR/du, dR/dv, dZ/du, dZ/dv
            test_ru = np.sum(test_rmnc[j, :] * (-sin_kernel) * xm)
            test_rv = np.sum(test_rmnc[j, :] * (-sin_kernel) * (-xn))
            test_zu = np.sum(test_zmns[j, :] * cos_kernel * xm)
            test_zv = np.sum(test_zmns[j, :] * cos_kernel * (-xn))

            ref_ru = np.sum(ref_rmnc[j, :] * (-sin_kernel) * xm)
            ref_rv = np.sum(ref_rmnc[j, :] * (-sin_kernel) * (-xn))
            ref_zu = np.sum(ref_zmns[j, :] * cos_kernel * xm)
            ref_zv = np.sum(ref_zmns[j, :] * cos_kernel * (-xn))

            kernel_nyq = xm_nyq * theta - xn_nyq * phi

            cos_kernel_nyq = np.cos(kernel_nyq)

            # B^z, B^t
            test_bsupu = np.sum(test_bsupumnc[j, :] * cos_kernel_nyq)
            test_bsupv = np.sum(test_bsupvmnc[j, :] * cos_kernel_nyq)

            ref_bsupu = np.sum(ref_bsupumnc[j, :] * cos_kernel_nyq)
            ref_bsupv = np.sum(ref_bsupvmnc[j, :] * cos_kernel_nyq)

            # B^r, B^phi, B^z
            test_br[l, k] = test_bsupu * test_ru + test_bsupv * test_rv
            test_bp[l, k] = test_bsupv * test_r[l, k]
            test_bz[l, k] = test_bsupu * test_zu + test_bsupv * test_zv

            ref_br[l, k] = ref_bsupu * ref_ru + ref_bsupv * ref_rv
            ref_bp[l, k] = ref_bsupv * ref_r[l, k]
            ref_bz[l, k] = ref_bsupu * ref_zu + ref_bsupv * ref_zv
    return (
        test_br,
        test_bp,
        test_bz,
        ref_br,
        ref_bp,
        ref_bz,
    )


def check_flux_surface_geometry(
    test_wout: netCDF4.Dataset, ref_wout: netCDF4.Dataset, all_vnv_checks: VnvChecks
) -> Status:
    for var in [
        "ns",
        "nfp",
        "xm",
        "xn",
        "xm_nyq",
        "xn_nyq",
        "rmnc",
        "zmns",
        "rmnc",
        "zmns",
        "bsupumnc",
        "bsupvmnc",
        "bsupumnc",
        "bsupvmnc",
    ]:
        # make sure none of these arrays is actually masked:
        # we want to cast them to unmasked arrays below.
        assert not ref_wout[var][:].mask.any()
        assert not test_wout[var][:].mask.any()

    ns = np.array(ref_wout["ns"][:]).item()
    nfp = np.array(ref_wout["nfp"][:]).item()

    xm = np.array(ref_wout["xm"][:])
    xn = np.array(ref_wout["xn"][:])

    xm_nyq = np.array(ref_wout["xm_nyq"][:])
    xn_nyq = np.array(ref_wout["xn_nyq"][:])

    ref_rmnc = np.array(ref_wout["rmnc"][:])
    ref_zmns = np.array(ref_wout["zmns"][:])

    test_rmnc = np.array(test_wout["rmnc"][:])
    test_zmns = np.array(test_wout["zmns"][:])

    ref_bsupumnc = np.array(ref_wout["bsupumnc"][:])
    ref_bsupvmnc = np.array(ref_wout["bsupvmnc"][:])

    test_bsupumnc = np.array(test_wout["bsupumnc"][:])
    test_bsupvmnc = np.array(test_wout["bsupvmnc"][:])

    # grid dimensions for real-space comparison
    # can be chosen as see fit...
    num_theta = 37
    num_phi = 36

    theta_grid = np.linspace(0.0, 2.0 * np.pi, num_theta, endpoint=False)
    phi_grid = np.linspace(0.0, 2.0 * np.pi / nfp, num_phi, endpoint=False)

    any_mismatch = False

    s = Status.OK

    test_wout_path = Path(test_wout.filepath())
    plots_dir = test_wout_path.parent / "diff_plots"
    conf_name = conf_name_from_path(test_wout_path)
    case_name = case_name_from_conf_name(conf_name)
    conf_checks = all_vnv_checks.checks[conf_name]

    var_suffixes = ("geometry_r", "geometry_z", "b_r", "b_phi", "b_z")

    # e.g. summary_stats["geometry_r"]["ref_mean"][j]
    summary_stats = {
        variable: {
            key: np.empty(ns)
            for key in ("ref_mean", "test_mean", "ref_stddev", "test_stddev")
        }
        for variable in var_suffixes
    }

    for j in range(ns):
        test_r, test_z, ref_r, ref_z = flux_surface_geom_rz(
            j, theta_grid, phi_grid, xm, xn, test_rmnc, test_zmns, ref_rmnc, ref_zmns
        )

        # cylindrical magnetic field components
        test_br, test_bp, test_bz, ref_br, ref_bp, ref_bz = flux_surface_geom_brbpbz(
            j,
            theta_grid,
            phi_grid,
            xm,
            xn,
            xm_nyq,
            xn_nyq,
            test_rmnc,
            test_zmns,
            ref_rmnc,
            ref_zmns,
            test_r,
            ref_r,
            test_bsupumnc,
            test_bsupvmnc,
            ref_bsupumnc,
            ref_bsupvmnc,
        )

        test_values = (test_r, test_z, test_br, test_bp, test_bz)
        ref_values = (ref_r, ref_z, ref_br, ref_bp, ref_bz)
        for var_suffix, val, ref in zip(var_suffixes, test_values, ref_values):
            summary_stats[var_suffix]["ref_mean"][j] = np.mean(ref)
            summary_stats[var_suffix]["ref_stddev"][j] = np.std(ref)
            summary_stats[var_suffix]["test_mean"][j] = np.mean(val)
            summary_stats[var_suffix]["test_stddev"][j] = np.std(val)

            var_suffix = f"flux_surface_{var_suffix}"
            tol = get_tolerance(var_suffix, case_name)
            unique_varname = f"{var_suffix}_j{j}"
            s = check_contents(val, ref, tol, varname=unique_varname)

            plot_path = Path("SEE SUMMARY PLOTS INSTEAD")
            conf_checks.append(
                ArrayCheck(
                    varname=unique_varname, tol=tol, status=s, diff_plot=plot_path
                )
            )
            if s is Status.MISMATCH:
                any_mismatch = True

    # produce summary errorbar plots
    for var_suffix in var_suffixes:
        plt.figure()
        plt.errorbar(
            x=np.arange(ns),
            y=summary_stats[var_suffix]["ref_mean"],
            yerr=summary_stats[var_suffix]["ref_stddev"],
        )
        plt.errorbar(
            x=np.arange(ns),
            y=summary_stats[var_suffix]["test_mean"],
            yerr=summary_stats[var_suffix]["test_stddev"],
        )
        figname = plots_dir / f"flux_surface_{var_suffix}_{conf_name}.SUMMARY.png"
        plt.savefig(figname)

    return Status.MISMATCH if any_mismatch else Status.OK


def compare_wout(
    *,
    vmecpp_wout: Path,
    reference_wout: Path,
    all_vnv_checks: VnvChecks,
) -> Status:
    """Compare the contents of two NetCDF3 wout files.

    Mismatches are logged and diff plots for 1D and 2D mismatched variables are stored
    in the directory diff_plots/ relative to vmecpp_wout.
    """

    vmecpp_ds = netCDF4.Dataset(vmecpp_wout)
    reference_ds = netCDF4.Dataset(reference_wout)

    plots_dir = vmecpp_wout.parent / "diff_plots"
    plots_dir.mkdir(exist_ok=True)

    overall_status = Status.OK

    for varname in [
        "niter",  # vnvchecklist.1.b
        # vnvchecklist.4.a (scalar wout quantities)
        "wb",
        "wp",
        "rmax_surf",
        "rmin_surf",
        "zmax_surf",
        "aspect",
        "betatotal",
        "betapol",
        "betator",
        "betaxis",
        "b0",
        "rbtor0",
        "rbtor",
        "IonLarmor",
        "volavgB",
        "ctor",
        "Aminor_p",
        "Rmajor_p",
        "volume_p",
        "fsqr",
        "fsqz",
        "fsql",
        "ier_flag",
        # vnvchecklist.4.b (radial profiles)
        "iotaf",
        "q_factor",
        "presf",
        "phi",
        "phipf",
        "chi",
        "chipf",
        "jcuru",
        "jcurv",
        "iotas",
        "mass",
        "pres",
        "beta_vol",
        "buco",
        "bvco",
        "vp",
        "specw",
        "phips",
        "over_r",
        "jdotb",
        "bdotgradv",
        "DMerc",
        "DShear",
        "DWell",
        "DCurr",
        "DGeod",
        "equif",
        # vnvchecklist.4.c (Fourier coefficients)
        "raxis_cc",
        "zaxis_cs",
        "rmnc",
        "zmns",
        "lmns",
        "bmnc",
        "gmnc",
        "bsubumnc",
        "bsubvmnc",
        "bsubsmns",
        "bsupumnc",
        "bsupvmnc",
    ]:
        s = check_shapes(vmecpp_ds[varname], reference_ds[varname])
        if s is Status.MISMATCH:
            overall_status = Status.MISMATCH
            continue

        case_name = case_name_from_conf_name(conf_name_from_path(vmecpp_wout))
        tol = get_tolerance(varname, case_name)

        val = vmecpp_ds[varname][:]
        ref = reference_ds[varname][:]
        s = check_contents(val, ref, tol, varname)
        if s is Status.MISMATCH:
            overall_status = Status.MISMATCH

        rank = len(val.shape)
        conf_name = vmecpp_wout.stem.removeprefix("wout_").removesuffix(".vmecpp")
        conf_checks = all_vnv_checks.checks[conf_name]
        if rank == 0:
            conf_checks.append(
                ScalarCheck(
                    varname=varname, test=val.item(), ref=ref.item(), tol=tol, status=s
                )
            )
        else:
            plot_path = plot_diffs(val, ref, varname, plots_dir, conf_name, s)
            conf_checks.append(
                ArrayCheck(varname=varname, tol=tol, status=s, diff_plot=plot_path)
            )

    s = check_flux_surface_geometry(vmecpp_ds, reference_ds, all_vnv_checks)
    if s is Status.MISMATCH:
        overall_status = Status.MISMATCH

    return overall_status


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "wout_under_test", help="Path to the wout_CONFIG.nc file under test.", type=Path
    )
    p.add_argument(
        "reference_wout", help="Path to the reference wout_CONFIG.nc file.", type=Path
    )
    args = p.parse_args()

    vmecpp_wout = args.wout_under_test
    ref_wout = args.reference_wout
    console.print(
        "\n\nComparing outputs:"
        f"\n\tunder test: {str(vmecpp_wout)}"
        f"\n\treference : {str(ref_wout)}"
    )

    conf_name = str(ref_wout).removeprefix("wout_").removesuffix(".nc")
    all_vnv_checks = VnvChecks(checks={conf_name: []})

    mismatches_found = compare_wout(
        vmecpp_wout=vmecpp_wout,
        reference_wout=ref_wout,
        all_vnv_checks=all_vnv_checks,
    )

    if mismatches_found:
        log_error("Mismatches found!", logger)
        sys.exit(1)
    console.print("All good!", style="green")

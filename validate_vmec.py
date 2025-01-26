# SPDX-FileCopyrightText: 2025-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
# /usr/bin/env python
#
# The driver for Proxima's VMEC++ Verification and Validation (V&V) procedure.
# Find the related spec at https://docs.google.com/document/d/1lfF_krpIQIFYQ6a5FH1odNd8PB1x5bH6KOslYy_Y8Dw/edit?usp=sharing
# Get usage information for this program by running `python validate_vmec.py --help`.

import argparse
import contextlib
import io
import itertools
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import NewType

import docker
import docker.errors
import numpy as np
import rich
import rich.console
import rich.progress as progress
from docker.models.containers import Container
from src._utils import (
    ScalarCheck,
    Status,
    VnvChecks,
    conf_name_from_path,
    log_error,
    log_info,
    log_ok,
)
from src.compare_wout import compare_wout
from src.input_generation import (
    EXCLUDED_CONFIGURATIONS,
    make_input_configs,
    make_input_configs_for_short_run,
)
from src.pdf_report import make_pdf_report
import vmecpp
from vmecpp.cpp.third_party.indata2json.indata_to_json import indata_to_json

logger = logging.getLogger("validate_vmec")
console = rich.console.Console()

RefConfPaths = NewType("RefConfPaths", list[Path])
VmecppConfPaths = NewType("VmecppConfPaths", list[Path])
RefWoutPaths = NewType("RefWoutPaths", list[Path])
VmecppWoutPaths = NewType("VmecppWoutPaths", list[Path])


def parse_args() -> argparse.Namespace:
    usage = """
    You might have to run `pants --tag=external_build run ::` to generate
    VMEC++'s Python bindings before running this script.

    RESULTS
    Besides producing human-readable output at the command line, this script
    creates a directory called 'vnvresults_YYYYMMDD_HHmm/' in the
    current working directory. The V&V log file, 'log.txt', is stored there.
    One sub-directory per test case will also be present, each containing:
    - Fortran VMEC2000 INDATA configuration files
    - VMEC++ JSON input configuration files derived from the Fortran INDATA files
    - 'wout' output files for the reference implementation and for VMEC++
      called 'wout_CONFIGNAME.nc' and 'wout_CONFIGNAME.vmecpp.nc' respectively
    - plots for any mismatched 1D and 2D quantities inside sub-directory 'diff_plots'

    If a directory with the same name as the results directory already exists,
    it is moved to $DIRNAME.old. If $DIRNAME.old also already exists, it is deleted.

    NOTES
    For debuggig purposes you can also call vmec_validation/compare_wout.py directly
    on a pair of wout files, see `python vmec_validation/compare_wout.py -h`.
    """

    p = ArgumentParser(
        description="Run Proxima's VMEC++ Verification and Validation procedure.",
        epilog=usage,
        formatter_class=RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--short",
        action="store_true",
        help=(
            "Run only on a subset of the input configurations, "
            "for quicker turnaround when testing"
        ),
    )
    p.add_argument(
        "--cache-dir",
        help=(
            "If present, use this directory as a cache space. "
            "Subsequent runs with the same cache directory might be able to skip "
            "certain expensive computations (e.g. the generation of wout files). "
            "NOTE: USE WITH CARE, there is no cache invalidation mechanism "
            "and in particular this script will not detect that the VMEC++ revision "
            "has changed and the wout files should be regenerated."
        ),
        type=lambda s: Path(s).resolve() if s is not None else None,
    )
    return p.parse_args()


def make_results_dir() -> Path:
    """Create results directory and return the corresponding Path.

    If the directory with the desired name already exists, it is first moved to
    $DIRNAME.old. If $DIRNAME.old already exists, it is deleted.
    """
    results_dir = Path(f"vnvresults_{datetime.now().strftime('%Y%m%d_%H%M')}")

    if results_dir.exists():  # move it to *.old
        bak_dir = results_dir.with_suffix(".old")

        if bak_dir.exists():
            shutil.rmtree(bak_dir)

        shutil.move(results_dir, bak_dir)

    results_dir.mkdir()
    return results_dir.absolute()


def configure_logger(logger: logging.Logger, results_dir: Path) -> None:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(results_dir / "log.txt", mode="w")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def log_header() -> None:
    banner = r"""
\ \   / ( _ )\ \   / /
 \ \ / // _ \/\ \ / /
  \ V /| (_>  <\ V /
   \_/  \___/\/ \_/
"""
    console.print(banner, overflow="crop", style="blue")
    logger.info("V&V started")


def make_vmecpp_confs(reference_confs: RefConfPaths) -> VmecppConfPaths:
    """Convert Fortran INDATA files to the corresponding JSON files using `indata2json`.

    Return the CONFIGNAME.json paths corresponding to each
    input.CONFIGNAME in reference_confs, in the same order.

    """
    all_vmecpp_confs = VmecppConfPaths([])
    for conf in reference_confs:
        original_cwd = Path.cwd()
        conf_dir = conf.parent

        # indata_to_json writes its output to the current working directory,
        # so we change there
        os.chdir(conf_dir)
        vmecpp_conf = indata_to_json(conf)
        os.chdir(original_cwd)

        all_vmecpp_confs.append(conf_dir / vmecpp_conf)

    return all_vmecpp_confs


class ReferenceWOutComputeTask:
    """A helper functor that represents the computation of one reference wout file."""

    docker_image: str
    results_dir: Path
    cache_dir: Path | None

    def __init__(self, docker_image: str, results_dir: Path, cache_dir: Path | None):
        self.docker_image = docker_image
        self.results_dir = results_dir
        self.cache_dir = cache_dir

    def __call__(self, conf_and_wout: tuple[Path, Path]) -> tuple[Path, str]:
        """Compute the reference wout file from the given input configuration running
        the reference Fortran VMEC2000 in Docker."""
        conf, wout = conf_and_wout

        wout.parent.mkdir(exist_ok=True)

        if self.cache_dir is not None:
            relative_wout_path = wout.relative_to(self.results_dir)
            cached_wout_path = self.cache_dir / relative_wout_path
            if cached_wout_path.exists():
                log_message = f"Found wout file '{relative_wout_path}' in the cache"
                shutil.copyfile(src=cached_wout_path, dst=wout)
                return (conf, log_message)

        # otherwise we actually have to run VMEC2000 (via docker)
        client = docker.from_env()
        container = client.containers.run(
            self.docker_image,
            command="tail -f /dev/null",  # a trick to keep the container running
            detach=True,
        )
        assert isinstance(container, Container)

        # copy input into container
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as tar:
            tar.add(conf, arcname=conf.name)

            # extract base conf name: input.w7x_beta0_mn5_ns25 -> w7x
            base_conf_name = conf_name_from_path(conf).split("_beta")[0]
            mgrid_file = conf.parent / f"mgrid_{base_conf_name}.nc"
            if mgrid_file.exists():
                tar.add(mgrid_file, arcname=mgrid_file.name)

        stream.seek(0)
        container.put_archive(path="/tmp", data=stream)

        vmec2000_exe = "/workdir/STELLOPT/VMEC2000/Release/xvmec2000"

        try:
            ret = container.exec_run(
                f"bash -c 'cd /tmp && {vmec2000_exe} {conf.name} && "
                f"chown {os.getuid()}:{os.getgid()} {wout.name}'"
            )
        except docker.errors.ContainerError as e:
            logs = bytes(container.logs()).decode("utf-8")
            msg = (
                f"Problem running VMEC2000 in Docker:\n\t{e}\n"
                f"Container logs:\n\t{logs}"
            )
            log_error(msg, logger)
            container.remove(force=True)
            raise RuntimeError(msg)

        logs = ret[1].decode("utf-8")
        if "error" in logs:
            msg = (
                f"Failure when running VMEC2000 on configuration '{conf}'. "
                f"Docker logs:\n\t{logs}"
            )
            log_error(msg, logger)
            raise RuntimeError(msg)

        # copy wout result from container
        bits, _ = container.get_archive(Path("/tmp") / wout.name)
        stream = BytesIO()
        for chunk in bits:
            stream.write(chunk)
        stream.seek(0)
        with tarfile.open(fileobj=stream) as tar:
            tar.extractall(path=wout.parent)

        container.stop()
        container.remove()

        if not wout.exists():
            msg = (
                "There was an error retrieving wout file from VMEC2000 container.\n"
                f"Configuration: {conf}"
                f"Expected wout file at: {wout}"
            )
            log_error(msg, logger)
            raise RuntimeError(msg)

        if self.cache_dir is not None:  # copy newly computed wout to cache directory
            cached_wout_path = self.cache_dir / wout.relative_to(self.results_dir)
            cached_wout_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src=wout, dst=cached_wout_path)

        return conf, logs


def extract_vmec2000_runtime(logs: str, conf: Path) -> float:
    if "TOTAL COMPUTATIONAL TIME" not in logs:
        msg = (
            "Could not find 'TOTAL COMPUTATIONAL TIME' in VMEC2000"
            f" logs for configuration {conf}"
        )
        log_error(msg, logger)
        return np.nan

    log_lines = logs.splitlines()
    total_runtime_line = next(
        line for line in log_lines if "TOTAL COMPUTATIONAL TIME" in line
    )
    assert total_runtime_line is not None

    # the line looks like:
    # TOTAL COMPUTATIONAL TIME               4.67 SECONDS
    # so the runtime is the fourth token (at index 3)
    runtime = float(total_runtime_line.split()[3])
    return runtime


def make_reference_wouts(
    confs: RefConfPaths, results_dir: Path, cache_dir: Path | None
) -> tuple[RefWoutPaths, list[float]]:
    """For every input.CONFIGNAME configuration, produce the corresponding
    wout_CONFIGNAME.nc file running Fortran VMEC.

    Return a list of paths to the wout files (in the same order as confs) as well as the
    list of runtimes for each of the Fortran VMEC runs.

    Note that running Fortran VMEC requires an internet connection to pull the latest
    version of the V&V Docker image.
    """

    def wout_path(conf: Path) -> Path:
        conf_name = conf_name_from_path(conf)
        ref_wout = conf.with_name(f"wout_{conf_name}.nc")
        return ref_wout

    wouts = RefWoutPaths([wout_path(c) for c in confs])
    runtimes = [-1.0] * len(confs)

    client = docker.from_env()
    try:
        client.ping()
    except docker.errors.APIError:
        msg = "Could not contact the Docker server, make sure Docker is up and running."
        raise RuntimeError(msg)

    with console.status("Pulling latest version of VMEC2000 Docker image..."):
        vnv_docker_img = "ghcr.io/proximafusion/vmec2000:latest"
        client.images.pull(vnv_docker_img)

    task = ReferenceWOutComputeTask(vnv_docker_img, results_dir, cache_dir)

    with progress.Progress(
        progress.SpinnerColumn(),
        progress.TextColumn("[progress.description]{task.description}"),
        progress.BarColumn(),
        progress.MofNCompleteColumn(),
        console=console,
    ) as progress_bar:
        task_id = progress_bar.add_task(
            "Generating reference wout files...", total=len(wouts)
        )
        # NOTE: we run VMEC2000 sequentially for two reasons:
        # - concurrent steering of docker containers from Python inccurred into
        #   some issues on at least one system (long container shutdown times and
        #   related errors)
        # - running many single-thread instances of VMEC++ concurrently on all cores
        #   distorts runtimes, so we run both VMEC2000 and VMEC++ sequentially
        # TODO(eguiraud): in case of errors in multiple sub-processes at the same
        # time, their logs and outputs might get garbled
        for conf, logs in map(task, zip(confs, wouts)):
            conf_idx = confs.index(conf)
            logs = logs.replace("\n", "\n\t")
            log_info(f"VMEC2000 logs for '{conf}':\n\t{logs}", logger, console)
            runtimes[conf_idx] = extract_vmec2000_runtime(logs, conf)
            progress_bar.advance(task_id)

    return wouts, runtimes


def run_vmecpp(conf_and_wout: tuple[Path, Path]) -> tuple[Path, str, float]:
    """Run VMEC++ and write the output at the path specified.

    Return a tuple containing the path to the configuration used for this run, the
    execution logs, and the runtime of this VMEC++ execution.
    """
    conf, wout = conf_and_wout
    out = StringIO()
    with contextlib.redirect_stdout(out):
        # with vmecpp.ostream_redirect(stdout=True, stderr=True):
        indata = vmecpp.VmecInput.from_file(conf)
        start_time = time.time()
        output_quantities = vmecpp.run(indata, max_threads=1)
        runtime = time.time() - start_time
        # fortran_wout = FortranWOutAdapter.from_vmecpp_wout(output_quantities.wout)
        output_quantities.wout.save(wout)
    return conf, out.getvalue(), runtime


def make_vmecpp_wouts(
    vmecpp_confs: VmecppConfPaths,
    results_dir: Path,
    cache_dir: Path | None,
) -> tuple[VmecppWoutPaths, list[float]]:
    """For every CONIGNAME.json configuration produce the corresponding
    'wout_CONFIGNAME.vmecpp.nc' file by running VMEC++.

    Return the paths to the wout files as a list with same ordering as vmecpp_confs, as
    well as a list of runtimes (also with same ordering as vmecpp_confs).
    """

    def wout_path(conf: Path):
        conf_name = conf_name_from_path(conf)
        wout_path = conf.with_name(f"wout_{conf_name}.vmecpp.nc")
        return wout_path

    wouts = VmecppWoutPaths([wout_path(c) for c in vmecpp_confs])
    runtimes = [np.nan] * len(wouts)

    if cache_dir is None:
        idxs_of_wouts_to_compute = list(range(len(wouts)))
    else:
        idxs_of_wouts_to_compute = []
        for i, wout_file in enumerate(wouts):
            relative_wout_path = wout_file.relative_to(results_dir)
            cached_wout_path = cache_dir / relative_wout_path
            if cached_wout_path.exists():
                log_info(f"Found file '{relative_wout_path}' in the cache", logger)
                shutil.copyfile(src=cached_wout_path, dst=wout_file)
            else:
                idxs_of_wouts_to_compute.append(i)

    with progress.Progress(
        progress.SpinnerColumn(),
        progress.TextColumn("[progress.description]{task.description}"),
        progress.BarColumn(),
        progress.MofNCompleteColumn(),
        console=console,
    ) as progress_bar:
        task_id = progress_bar.add_task(
            "Generating VMEC++ wout files...", total=len(idxs_of_wouts_to_compute)
        )
        try:
            # NOTE: here we run VMEC++ single-threaded, sequentially for each
            # configuration, which is very inefficient compared to dispatching
            # the runs in parallel on a multi-processing pool or similar.
            # However, saturating all cores with VMEC++ runs results in a runtime
            # performance degradation of many runs, and we care about runtimes as
            # part of the comparison with the reference implementation.
            # TODO(eguiraud); check whether we can fill e.g. 50% or 25% of the cores,
            # recovering some parallelism, without significant runtime degradation.
            for conf, out, runtime in map(
                run_vmecpp,
                ((vmecpp_confs[i], wouts[i]) for i in idxs_of_wouts_to_compute),
            ):
                conf_idx = vmecpp_confs.index(conf)
                runtimes[conf_idx] = runtime
                log_info(f"VMEC++ logs for '{conf}':\n{out}", logger, console=console)
                progress_bar.advance(task_id)

                if cache_dir is not None:  # copy newly computed wout to cache directory
                    wout = wouts[conf_idx]
                    cached_wout = cache_dir / wout.relative_to(results_dir)
                    cached_wout.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src=wout, dst=cached_wout)
        except RuntimeError as e:  # vnvchecklist.1.a
            log_error(f"Failure while running VMEC++: {e}", logger)
            raise e

    return wouts, runtimes


def check_runtimes(
    confs: RefConfPaths,
    *,
    vmecpp_runtimes: list[float],
    reference_runtimes: list[float],
    all_vnv_checks: VnvChecks,
) -> Status:
    overall_status = Status.OK
    tol = 1.1  # we are ok with a 10% slowdown
    for vmecpp_time, ref_time, conf in zip(vmecpp_runtimes, reference_runtimes, confs):
        conf_name = conf_name_from_path(conf)
        s = Status.OK
        # vnvchecklist.2
        if np.isnan(vmecpp_time) or np.isnan(ref_time):
            msg = (
                "Could not retrieve runtimes for VMEC++ and/or VMEC2000 "
                f"for configuration '{str(conf)}' (maybe the wout file was cached?)"
            )
            log_error(msg, logger)
            s = Status.MISMATCH
            overall_status = Status.MISMATCH
        if vmecpp_time > ref_time * tol and vmecpp_time > ref_time + 1.0:
            msg = (
                "VMEC++ runtime is more than 10% "
                "and more than 1 sec longer than the reference's "
                f"for configuration '{str(conf)}':\n"
                f"VMEC++: {vmecpp_time}\nVMEC2000: {ref_time}"
            )
            log_error(msg, logger)
            s = Status.MISMATCH
            overall_status = Status.MISMATCH
        all_vnv_checks.checks[conf_name].append(
            ScalarCheck(
                varname="runtime",
                test=vmecpp_time,
                ref=ref_time,
                tol=tol,
                status=s,
            )
        )
    return overall_status


def log_status(status: Status) -> None:
    if status is Status.MISMATCH:
        msg = "Mismatches were found!"
        log_error(msg, logger)
    else:
        log_ok("All good!", logger)


def log_footer() -> None:
    logger.info("V&V finished")


def check_all_checks_are_present(
    checks: VnvChecks, reference_confs: list[Path], short: bool
) -> None:
    """Log an error if all_vnv_checks is somehow missing some entries that are expected.

    We do not want to raise an error that might terminate execution: it
    is useful to produce an output even if it is partial.
    """
    log_info("Checking that the summary file contains all expected checks...", logger)

    if short:
        # in a short run, the conf_names are whatever they are
        conf_names = list(map(conf_name_from_path, reference_confs))
    else:
        # otherwise they must be what the spec dictates
        # NOTE: solovev_analytical, SQuID_v1, CTH-like, W7-X, NCSX and PARVMEC_fails
        # are not implemented yet
        configs = [
            "dshape",
            "heliotron",
            "cm_a",
            "cm_b",
            "precise_qa",
            "precise_qh",
        ]
        betas = [0, 1, 5]
        mns = [5, 7, 12]
        nss = [25, 51, 99]
        conf_names = []
        for case_name in configs:
            for beta, mn, ns in itertools.product(betas, mns, nss):
                if (case_name, beta, mn, ns) not in EXCLUDED_CONFIGURATIONS:
                    conf_names.append(f"{case_name}_beta{beta}_mn{mn}_ns{ns}")

    # check keys
    # iterate on a copy because elements are removed as we go, which breaks iteration
    for conf_name in conf_names.copy():
        if conf_name not in checks.checks:
            log_error(f"Missing key in checks JSON: '{conf_name}'", logger)
            conf_names.remove(conf_name)

    # check contents
    def log_missing(varname: str, conf_name: str):
        log_error(
            f"Missing key in checks JSON: '{varname}' for configuration '{conf_name}'",
            logger,
        )

    for conf_name in conf_names:
        checked_vars = [c.varname for c in checks.checks[conf_name]]

        # runtimes (vnvchecklist.2)
        if "runtime" not in checked_vars:
            log_missing(varname="runtime", conf_name=conf_name)

        # flux surface geometry (vnvchecklist.3)
        key_suffixes = ["geometry_r", "geometry_z", "b_r", "b_phi", "b_z"]
        for k in key_suffixes:
            # we check only for j == 0, assuming if that is present the values for
            # all surfaces will be present
            varname = f"flux_surface_{k}_j0"
            if varname not in checked_vars:
                log_missing(varname, conf_name)

        # wout contents
        wout_variables = [
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
        ]
        for varname in wout_variables:
            if varname not in checked_vars:
                log_missing(varname, conf_name)

        # post-optimization quantities, only for higher-resolution configurations
        re_match = re.match(r"\w+_beta(\d+)_mn(\d+)_ns(\d+)", conf_name)
        assert re_match is not None
        beta, mn, ns = map(int, re_match.groups())
        if not (beta == 5 and mn == 12 and ns == 99):
            continue

        post_opt_varnames = [
            "cobravmec_growth_rates",  # vnvchecklist.8
            "lost_fraction",  # vnvchecklist.9
        ]
        for varname in post_opt_varnames:
            if varname not in checked_vars:
                log_missing(varname, conf_name)

        # NOTE: SFINCS and spiderplot (vnvchecklist.10-11) not implemented
        # NOTE: "green CI check" (vnvchecklist.12) not implemented


def main() -> int:
    results_dir = make_results_dir()
    configure_logger(logger, results_dir)
    log_header()

    args = parse_args()

    log_info("Generating configurations (VMEC input files and mgrid files)...", logger)

    if args.short:
        reference_confs = RefConfPaths(
            make_input_configs_for_short_run(results_dir, cache_dir=args.cache_dir)
        )
    else:
        reference_confs = RefConfPaths(
            make_input_configs(results_dir, cache_dir=args.cache_dir)
        )

    log_info(
        (
            f"Generated {len(reference_confs)} configurations: "
            f"{list(map(str, reference_confs))}"
        ),
        logger,
    )

    log_info("Producing VMEC++ JSON input configurations...", logger)
    vmecpp_confs = make_vmecpp_confs(reference_confs)
    log_ok(f"{len(vmecpp_confs)} VMEC++ JSON input configurations generated.", logger)

    log_info("Producing reference wout files...", logger)
    reference_wouts, reference_runtimes = make_reference_wouts(
        reference_confs, results_dir, cache_dir=args.cache_dir
    )
    log_ok("All reference wout files have been successfully generated.", logger)

    log_info("Producing VMEC++ wout files...", logger)
    vmecpp_wouts, vmecpp_runtimes = make_vmecpp_wouts(
        vmecpp_confs, results_dir, cache_dir=args.cache_dir
    )
    log_ok("All VMEC++ wout files have been successfully generated.", logger)

    all_vnv_checks = VnvChecks.from_conf_paths(reference_confs)

    log_info("Comparing runtimes...", logger)
    any_mismatch = check_runtimes(
        reference_confs,
        vmecpp_runtimes=vmecpp_runtimes,
        reference_runtimes=reference_runtimes,
        all_vnv_checks=all_vnv_checks,
    )
    log_info("Finished comparing runtimes", logger)

    for vmecpp_wout, ref_wout in zip(vmecpp_wouts, reference_wouts):
        log_info(
            (
                "Comparing contents of wout files:"
                f"\n\tunder test: {str(vmecpp_wout)}"
                f"\n\treference : {str(ref_wout)}"
            ),
            logger,
        )
        status = compare_wout(
            vmecpp_wout=vmecpp_wout,
            reference_wout=ref_wout,
            all_vnv_checks=all_vnv_checks,
        )

        log_status(status)
        if status is Status.MISMATCH:
            any_mismatch = True

    check_all_checks_are_present(all_vnv_checks, reference_confs, args.short)

    summary_fname = "all_vnv_checks.json"
    with open(results_dir / summary_fname, "w") as f:
        f.write(all_vnv_checks.model_dump_json(indent=2))
    log_info(f"V&V summary written to {summary_fname}", logger)

    make_pdf_report(all_vnv_checks, results_dir)

    log_footer()
    return 1 if any_mismatch else 0


if __name__ == "__main__":
    sys.exit(main())

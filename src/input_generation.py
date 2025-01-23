# Generate input configurations for VMEC++'s V&V process
# Implements the spec at https://docs.google.com/document/d/1vM_y6-33-pDqdLGccfGxM84Y-NynWhW74UAfOqQXWVA/edit?usp=sharing.

import itertools
import logging
import os
import shutil
import subprocess
from pathlib import Path

from src._utils import log_error, log_info

logger = logging.getLogger("validate_vmec.input_generation")

# Mapping (case_name, beta) -> pres_scale
PRES_SCALES_PER_BETA: dict[tuple[str, int], float] = {
    ("cm_a", 1): 1.5e3,
    ("cm_a", 5): 7.5e3,
    ("cm_b", 1): 11.0e3,
    ("cm_b", 5): 55.0e3,
    ("dshape", 1): 0.6e3,
    ("dshape", 5): 2.7e3,
    ("heliotron", 1): 2.0e3,
    ("heliotron", 5): 9.0e3,
    ("precise_qa", 1): 8.5e3,
    ("precise_qa", 5): 42.0e3,
    ("precise_qh", 1): 9.0e3,
    ("precise_qh", 5): 41.0e3,
    ("ncsx", 1): 0.27,
    ("ncsx", 5): 1.2,
    ("w7x", 1): 65.0e3,
    ("w7x", 5): 305.0e3,
    ("cth_like", 1): 2.2e3,
}

# A list of configurations that are known to not converge with VMEC2000.
# They are excluded from the parameter sweep
EXCLUDED_CONFIGURATIONS = (
    # CASE, BETA, MN, NS
    ("cm_a", 0, 12, 99),
    ("cm_a", 1, 12, 99),
    ("cm_a", 5, 7, 99),
    ("cm_a", 5, 12, 25),
    ("cm_a", 5, 12, 51),
    ("cm_a", 5, 12, 99),
    # ncsx with mn==12 never converges
    ("ncsx", 0, 12, 25),
    ("ncsx", 1, 12, 25),
    ("ncsx", 5, 12, 25),
    ("ncsx", 0, 12, 51),
    ("ncsx", 1, 12, 51),
    ("ncsx", 5, 12, 51),
    ("ncsx", 0, 12, 99),
    ("ncsx", 1, 12, 99),
    ("ncsx", 5, 12, 99),
    # cth_like with beta==5 never converges
    ("cth_like", 5, 5, 25),
    ("cth_like", 5, 5, 51),
    ("cth_like", 5, 5, 99),
    ("cth_like", 5, 7, 25),
    ("cth_like", 5, 7, 51),
    ("cth_like", 5, 7, 99),
    ("cth_like", 5, 12, 25),
    ("cth_like", 5, 12, 51),
    ("cth_like", 5, 12, 99),
)


def make_mgrid_file(
    case_name: str, target_folder: Path, cache_dir: Path | None
) -> None:
    mgrid_fname = f"mgrid_{case_name}.nc"
    output_mgrid_file = target_folder / case_name / mgrid_fname

    if output_mgrid_file.exists():
        return  # mgrid file already there for this case name

    if cache_dir is None or not (cache_dir / case_name / mgrid_fname).exists():
        mgrid_exe = shutil.which("mgrid")
        if mgrid_exe is None:
            msg = "Could not find 'mgrid' executable in PATH."
            log_error(msg, logger)
            raise RuntimeError(msg)

        coils_file = Path(__file__).parent / "coils" / f"coils.{case_name}"

        status = subprocess.run(
            [mgrid_exe, coils_file],
            cwd=coils_file.parent,
            capture_output=True,
        )
        if status.returncode != 0 or "error" in status.stderr.decode().lower():
            msg = (
                f"There was an issue running command '{mgrid_exe} {coils_file}'.\n"
                f"stdout:\n{status.stdout}"
                f"stderr:\n{status.stderr}"
            )
            log_error(msg, logger)
            raise RuntimeError(msg)

        if not (coils_file.parent / mgrid_fname).exists():
            msg = (
                f"Command '{mgrid_exe} {coils_file}' did not "
                "produce file {mgrid_fname}.\n"
            )
            log_error(msg, logger)
            raise RuntimeError(msg)

        src_file = coils_file.parent / mgrid_fname
        shutil.move(src_file, output_mgrid_file)
        if not output_mgrid_file.exists():
            msg = f"Failed to move '{src_file}' to '{output_mgrid_file}'.\n"
            log_error(msg, logger)
            raise RuntimeError(msg)

        if cache_dir is not None:
            cached_mgrid_file_path = cache_dir / case_name / mgrid_fname
            cached_mgrid_file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src=output_mgrid_file, dst=cached_mgrid_file_path)
    else:  # file already in the cache
        log_info(f"Found file '{Path(case_name, mgrid_fname)}' in the cache", logger)
        cached_mgrid_file_path = cache_dir / case_name / mgrid_fname
        shutil.copyfile(src=cached_mgrid_file_path, dst=output_mgrid_file)


def make_input(
    case_name: str,
    target_folder: Path,
    *,
    beta: int,
    mn: int,
    ns: int,
    cache_dir: Path | None,
) -> Path:
    indata_path = Path(case_name, f"input.{case_name}_beta{beta}_mn{mn}_ns{ns}")
    output_fname = target_folder / indata_path

    if cache_dir is None or not (cache_dir / indata_path).exists():
        # produce indata file
        if beta == 0:
            pres_scale = 0.0
        else:
            pres_scale = PRES_SCALES_PER_BETA[case_name, beta]

        if case_name == "cm_a":
            mpol = mn
            ntor = mpol + 1 if mpol < 10 else mpol + 2
        elif case_name == "dshape":
            mpol = mn
            ntor = 0
        elif case_name == "cth_like":
            mpol = mn + 1
            ntor = mn
        else:
            mpol = mn
            ntor = mpol

        template_file = Path(__file__).parent / "indata_templates" / f"{case_name}.txt"
        with open(template_file) as f:
            template = f.read()
            if case_name == "cth_like":
                ftol = 1.0e-11 if ns > 50 else 1.0e-12
                config = template.format(
                    ns=ns, mpol=mpol, ntor=ntor, pres_scale=pres_scale, ftol=ftol
                )
            else:
                config = template.format(
                    ns=ns, mpol=mpol, ntor=ntor, pres_scale=pres_scale
                )

        with open(output_fname, "w") as f:
            f.write(config)

        if cache_dir is not None:  # store the result in the cache directory
            cached_indata_file = cache_dir / indata_path
            cached_indata_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src=output_fname, dst=cached_indata_file)
    else:  # file already in cache
        log_info(f"Found file '{indata_path}' in the cache", logger)
        shutil.copyfile(src=cache_dir / indata_path, dst=output_fname)

    # produce mgrid file (only free-boundary configurations)
    if case_name in ["w7x", "ncsx", "cth_like"]:
        make_mgrid_file(case_name, target_folder, cache_dir=cache_dir)

    return output_fname.absolute()


def make_input_configs_for_short_run(
    results_dir: Path, cache_dir: Path | None
) -> list[Path]:
    """Return a list of absolute paths to a subset of the V&V input configurations.

    Useful for quicker turnaround.
    """

    for case_name in ["dshape", "heliotron",]:
        os.makedirs(results_dir / case_name)

    confs = [
        make_input("dshape", results_dir, beta=1, mn=7, ns=51, cache_dir=cache_dir),
        make_input("heliotron", results_dir, beta=5, mn=7, ns=51, cache_dir=cache_dir),
    ]

    return confs


def make_input_configs(results_dir: Path, cache_dir: Path | None) -> list[Path]:
    """Return a list of absolute paths to the input configurations."""

    betas = [0, 1, 5]
    mns = [5, 7, 12]
    nss = [25, 51, 99]
    configs = [
        "dshape",
        "heliotron",
        "cm_a",
        "cm_b",
        "precise_qa",
        "precise_qh",
        "ncsx",
        "w7x",
        "cth_like",
    ]

    all_reference_configs: list[Path] = []
    for case_name in configs:
        os.makedirs(results_dir / case_name)
        for beta, mn, ns in itertools.product(betas, mns, nss):
            if (case_name, beta, mn, ns) in EXCLUDED_CONFIGURATIONS:
                continue
            conf = make_input(
                case_name, results_dir, beta=beta, mn=mn, ns=ns, cache_dir=cache_dir
            )
            all_reference_configs.append(conf)

    return all_reference_configs

import logging
from enum import Enum
from pathlib import Path

import numpy as np
import rich.box
import rich.console
import rich.table
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger("validate_vmec.utils")
logger.addHandler(logging.NullHandler())
console = rich.console.Console()


class Status(Enum):
    OK = "OK"
    MISMATCH = "MISMATCH"


class ScalarCheck(BaseModel):
    # to configure pydantic to read/write nans
    # from https://github.com/pydantic/pydantic/issues/7007
    model_config = ConfigDict(ser_json_inf_nan="constants")  # pyright: ignore
    varname: str
    test: float
    ref: float
    tol: float
    status: Status


class ArrayCheck(BaseModel):
    varname: str
    tol: float
    status: Status
    diff_plot: Path  # path to a file containing the diff plot for this variable


# A type to store the results of all V&V checks.
# checks = {"conf_name": [ScalarCheck, ArrayCheck, ...] }
class VnvChecks(BaseModel):
    checks: dict[str, list[ScalarCheck | ArrayCheck]]

    @staticmethod
    def from_conf_paths(confs: list[Path]) -> "VnvChecks":
        return VnvChecks(checks={conf_name_from_path(c): [] for c in confs})


def conf_name_from_path(conf: Path) -> str:
    """path/to/(input.,wout_)conf_name[.json] -> conf_name."""
    return (
        # cannot use `stem` because the stem of `input.blah` is `input`
        conf.name.removeprefix("input.")
        .removeprefix("wout_")
        .removesuffix(".json")
        .removesuffix(".nc")
        .removesuffix(".vmecpp")
    )


def case_name_from_conf_name(conf_name: str) -> str:
    """`conf_name` is in the format returned by `conf_name_from_path` and it could be
    e.g. w7x_beta0_mn12_ns51.

    The corresponding case name returned will be w7x.
    """
    return conf_name.split("_beta")[0]


def relabs_errors(
    actual: Float[np.ndarray, "..."], expected: Float[np.ndarray, "..."]
) -> Float[np.ndarray, "..."]:
    errors = (actual - expected) / (1.0 + np.abs(expected))
    return errors


def log_info(msg: str, logger: logging.Logger, console=console) -> None:
    """Pass in a console object if it's important that it is that specific one that is
    used for printing, e.g. when attaching a console to a progress bar."""
    console.print("[bold][INFO][/bold]", msg)
    logger.info(msg)


def log_ok(msg: str, logger: logging.Logger) -> None:
    console.print("[bold green][OK][/bold green]", msg)
    logger.info(f"[OK] {msg}")


def log_error(msg: str, logger: logging.Logger) -> None:
    console.print("[bold red][ERROR][/bold red]", msg)
    logger.error(msg)


def check_contents(
    test_var: Float[np.ndarray, "..."],
    ref_var: Float[np.ndarray, "..."],
    tol: float,
    varname: str,
) -> Status:
    if np.any(np.isnan(test_var)) or np.any(np.isnan(ref_var)):
        # if any value is NaN, something went particularly wrong
        msg = (
            f"There was an error when evaluating '{varname}':"
            f"\n\tunder test: {test_var}"
            f"\n\treference : {ref_var}"
        )
        log_error(msg, logger)
        return Status.MISMATCH

    errors = relabs_errors(test_var, ref_var)

    mismatch_mask = np.abs(errors) > tol
    mismatch_found = np.any(mismatch_mask)
    if not mismatch_found:
        return Status.OK

    if test_var.shape == ():
        msg = (
            f"Mismatch in '{varname}' (tolerance: {tol}):"
            f"\n\tunder test: {test_var:.16e}"
            f"\n\treference : {ref_var:.16e}"
            f"\n\terror     : {errors.item():.16e}"
        )
        log_error(msg, logger)
    else:
        wrong_idxs = np.nonzero(mismatch_mask)
        table = rich.table.Table(
            title_justify="left",
            box=rich.box.SIMPLE,
        )
        table.add_column("idx")
        table.add_column("under test\nreference", justify="right")
        table.add_column("error", justify="right")
        for idx in zip(*wrong_idxs):
            table.add_row(
                f"{idx}",
                f"{test_var[idx]:.16e}\n{ref_var[idx]:.16e}",
                f"{errors[idx]:e}",
            )
        log_error(f"Mismatch in '{varname}' (tolerance: {tol})", logger)
        c = rich.console.Console(record=True)
        c.print(table)
        logger.error(c.export_text())

    return Status.MISMATCH

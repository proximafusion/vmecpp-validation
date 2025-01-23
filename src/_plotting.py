import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float
from src._utils import Status

logger = logging.getLogger("validate_vmec.plotting")
logger.addHandler(logging.NullHandler())


def plot_diffs(
    val: Float[np.ndarray, "..."],
    ref: Float[np.ndarray, "..."],
    varname: str,
    plots_dir: Path,
    conf_name: str,
    status: Status,
) -> Path:
    rank = len(val.shape)
    if rank == 1:
        return _plot_diffs_1d(val, ref, varname, plots_dir, conf_name, status)
    elif rank == 2:
        return _plot_diffs_2d(val, ref, varname, plots_dir, conf_name, status)
    else:
        raise RuntimeError("Cannot plot variable with rank different from 1 or 2")


def _plot_diffs_1d(
    val: Float[np.ndarray, "..."],
    ref: Float[np.ndarray, "..."],
    varname: str,
    plots_dir: Path,
    conf_name: str,
    status: Status,
) -> Path:
    status_str = "OK" if status is Status.OK else "ERROR"
    _, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].plot(ref, "bo-", label="ref")
    ax[0].plot(val, "rx--", label="tst")
    ax[0].set_title(f"{varname} ({status_str})")
    ax[0].legend(loc="upper center")
    ax[0].grid(True)
    errProfile = (val - ref) / (1.0 + np.abs(ref))
    ax[1].plot(errProfile, "k.-")
    ax[1].grid(True)
    fname = plots_dir / f"{varname}_{conf_name}.{status_str}.png"
    plt.savefig(fname)
    plt.close()
    return fname


def _plot_diffs_2d(
    val: Float[np.ndarray, "..."],
    ref: Float[np.ndarray, "..."],
    varname: str,
    plots_dir: Path,
    conf_name: str,
    status: Status,
) -> Path:
    status_str = "OK" if status is Status.OK else "ERROR"
    errProfile = (val - ref) / (1.0 + np.abs(ref))
    plt.subplot(1, 3, 1)
    plt.pcolormesh(ref, cmap="jet")
    plt.xlabel("ref")
    plt.subplot(1, 3, 2)
    plt.pcolormesh(val, cmap="jet")
    plt.colorbar()
    plt.xlabel("tst")
    plt.title(f"{varname} ({status_str})")
    plt.subplot(1, 3, 3)
    plt.pcolormesh(errProfile, cmap="jet")
    plt.colorbar()
    plt.xlabel("err")
    plt.tight_layout()
    fname = plots_dir / f"{varname}_{conf_name}.{status_str}.png"
    plt.savefig(fname)
    plt.close()
    return fname

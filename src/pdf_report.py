# SPDX-FileCopyrightText: 2025-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
import os
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from src._utils import ArrayCheck, ScalarCheck, Status, VnvChecks


def get_mismatch_report(check: ScalarCheck | ArrayCheck) -> str:
    output = (
        f"**`{check.varname}`**...\\textcolor{{red}}{{MISMATCH}} (tol: {check.tol})\n"
    )
    if isinstance(check, ScalarCheck):
        output += "```\n"
        output += f"    under test: {check.test}\n"
        output += f"    reference : {check.ref}\n"
        output += "```\n"
    else:
        assert isinstance(check, ArrayCheck)
        output += f"\\\n![]({check.diff_plot})\\\n"
    return output


def get_conf_report(
    conf_name: str, checks: list[ScalarCheck | ArrayCheck], has_error: bool
) -> str:
    output = f"\n## `{conf_name}`: "
    if has_error:
        output += "\\textcolor{red}{errors found}\n"
    else:
        output += "all good\n"

    good_checks = 0
    has_bad_checks = False
    for check in checks:
        if check.status is Status.OK:
            good_checks += 1
        else:
            has_bad_checks = True
            output += get_mismatch_report(check)
    if good_checks > 0:
        maybe_other = "other " if has_bad_checks else ""
        output += f"{good_checks} {maybe_other}checks were OK.\\\n"

    return output


def make_pdf_report(all_vnv_checks: VnvChecks, output_dir: Path) -> None:
    checks = all_vnv_checks.checks

    output = f"# V&V results, {datetime.now().strftime('%d/%m/%Y %H:%M')}\\\n"

    def has_errors(checks_list: list[ScalarCheck | ArrayCheck]) -> bool:
        return any(c.status is Status.MISMATCH for c in checks_list)

    # first all the configurations with errors...
    confs_with_errors = [conf for conf in checks.keys() if has_errors(checks[conf])]
    for conf in confs_with_errors:
        conf_str = get_conf_report(conf, checks[conf], has_error=True)
        output += conf_str

    # ...then all the other ones
    confs_without_errors = [
        conf for conf in checks.keys() if conf not in confs_with_errors
    ]
    for conf in confs_without_errors:
        conf_str = get_conf_report(conf, checks[conf], has_error=False)
        output += conf_str

    # write out markdown file
    output_path = output_dir / "report.md"
    with open(output_path, "w") as f:
        f.write(output)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "vnv_json_summary",
        help="JSON file that contains a summary of all V&V checks",
        type=Path,
    )
    args = p.parse_args()

    with open(args.vnv_json_summary) as f:
        checks = VnvChecks.model_validate_json(f.read())

    make_pdf_report(checks, Path(os.getcwd()))

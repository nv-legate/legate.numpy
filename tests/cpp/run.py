# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import argparse
import os
import subprocess
import sys

LAUNCHER_VAR_PREFIXES = (
    "CONDA_",
    "LEGATE_",
    "LEGION_",
    "LG_",
    "REALM_",
    "GASNET_",
    "PYTHON",
    "UCX_",
    "NCCL_",
    "CUNUMERIC_",
    "NVIDIA_",
)

test_args_dict = {
    # Example of usage
    # "Alignment.Basic" : ["-logfile", "build/example_file.log"]
}


def fetch_test_names(binary_path):
    list_command = [binary_path] + ["--gtest_list_tests"]

    result = subprocess.check_output(list_command, stderr=subprocess.STDOUT)
    result = result.decode(sys.stdout.encoding).split("\n")

    test_group = ""
    test_names = []
    for line in result:
        # Skip empty entry
        if not line.strip():
            continue

        # Check if this is a test group
        if line[0] != " ":
            test_group = line.strip()
            continue

        # Assign test to test group
        test_names += [test_group + line.strip()]

    return test_names


def run_test(config, test_name, log, extra_args):
    test_command = []
    if config.ranks != 0:
        test_command += ["mpirun", "-n", str(config.ranks)]
        test_command += ["--output-filename", "build/mpi_result"]
        test_command += ["--merge-stderr-to-stdout"]

        def is_launcher_var(name: str) -> bool:
            # Whether an environment variable name is relevant for the laucher
            return name.endswith("PATH") or any(
                name.startswith(prefix) for prefix in LAUNCHER_VAR_PREFIXES
            )

        for var in dict(os.environ):
            if is_launcher_var(var):
                test_command += ["-x", var]

    test_command += [config.binary_path]
    test_command += [f"--gtest_filter={test_name}"]
    test_command += ["-ll:cpu", str(config.cpus)]
    test_command += extra_args

    if test_name in test_args_dict:
        test_command += test_args_dict[test_name]

    task = subprocess.Popen(test_command, stdout=log, stderr=subprocess.STDOUT)
    task.communicate()

    return task.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Legate cpp tests.")
    parser.add_argument(
        "--binary-path",
        dest="binary_path",
        required=False,
        default="build/cpp_tests",
        help="Path to binary under test.",
    )
    parser.add_argument(
        "--log-path",
        dest="log_path",
        required=False,
        default="build/results.log",
        help="Path to output log file.",
    )
    parser.add_argument(
        "--ranks",
        dest="ranks",
        required=False,
        type=int,
        default=0,
        help="Runs mpirun with rank if non-zero.",
    )
    parser.add_argument(
        "--cpus",
        dest="cpus",
        required=False,
        type=int,
        default=4,
        help="Legion cmd argument for CPU processors to create per process.",
    )
    config, extra_args = parser.parse_known_args()

    # Get names
    test_names = fetch_test_names(config.binary_path)

    # Run each test with popen
    total_count = len(test_names)
    failed_count = 0
    failed_tests = []
    with open(config.log_path, "w") as log:
        for count, test_name in enumerate(test_names):
            return_code = run_test(config, test_name, log, extra_args)

            # Record test result
            if return_code:
                failed_tests += [test_name]
                failed_count += 1
            print(
                f"{count+1:3d}/{total_count}: {test_name} ".ljust(50, "."),
                "Failed" if return_code else "Passed",
            )

    # Summarize results
    print(
        f"\n{int((total_count - failed_count) / total_count * 100)}% "
        f"tests passed, {failed_count} tests failed out of {total_count}"
    )
    if failed_tests:
        print("\nThe following tests FAILED:")
        for test in failed_tests:
            print(f"    - {test} (Failed)")
        print(f"\nLog file generated: {config.log_path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

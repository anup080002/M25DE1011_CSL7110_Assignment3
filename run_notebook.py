from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def execute_notebook(notebook_path: Path, kernel_name: str, timeout: int) -> None:
    nb = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=kernel_name,
        allow_errors=False,
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()
    nbformat.write(nb, notebook_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the assignment notebook in place.")
    parser.add_argument(
        "--notebook",
        default="recommender_systems_assignment_solution.ipynb",
        help="Path to the notebook to execute.",
    )
    parser.add_argument(
        "--kernel",
        default="python3",
        help="Kernel name to use for execution.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="Cell execution timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook_path = Path(args.notebook).resolve()
    execute_notebook(notebook_path, kernel_name=args.kernel, timeout=args.timeout)
    print(f"Executed notebook: {notebook_path}")


if __name__ == "__main__":
    main()

"""The provenance-api command: operational tasks that are not HTTP endpoints."""

import argparse
import asyncio
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from provenance_api.config import BackendSettings
from provenance_api.db import create_engine, create_session_factory
from provenance_api.ml import nltk_data
from provenance_api.ml.train import ARTIFACT_DIR, DEFAULT_DATASET, train, write_artifacts
from provenance_api.models import Plan, Role
from provenance_api.set_role import set_plan, set_role
from provenance_api.worker import create_worker


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand and return the process exit code."""
    args = _build_parser().parse_args(argv)
    exit_code: int = args.handler(args)
    return exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="provenance-api", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser(
        "train", help="Retrain the classifier and write the model files."
    )
    train_parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACT_DIR,
        help="Where to write model.joblib and model.json. Defaults to the packaged location.",
    )
    train_parser.set_defaults(handler=_train)

    data_parser = commands.add_parser(
        "nltk-data", help="Download the NLTK data files feature extraction needs."
    )
    data_parser.add_argument(
        "target_directory", nargs="?", help="Defaults to NLTK's own data directory."
    )
    data_parser.set_defaults(handler=_nltk_data)

    role_parser = commands.add_parser(
        "set-role", help="Change a user's role. This is how the first admin is made."
    )
    role_parser.add_argument("email")
    role_parser.add_argument("role", type=Role, choices=list(Role))
    role_parser.set_defaults(handler=_set_role)

    plan_parser = commands.add_parser(
        "set-plan", help="Change a user's plan by hand, without going through billing."
    )
    plan_parser.add_argument("email")
    plan_parser.add_argument("plan", type=Plan, choices=list(Plan))
    plan_parser.set_defaults(handler=_set_plan)

    worker_parser = commands.add_parser(
        "worker", help="Run the background worker that scores batch jobs."
    )
    worker_parser.set_defaults(handler=_worker)

    return parser


def _train(args: argparse.Namespace) -> int:
    if not args.dataset.is_file():
        print(f"Dataset not found: {args.dataset}")
        return 1
    result = train(args.dataset)
    model_path = write_artifacts(result, args.output_dir)
    print(json.dumps(result.metadata["metrics"], indent=2))
    print(f"Wrote {model_path} ({result.metadata['model_version']})")
    return 0


def _nltk_data(args: argparse.Namespace) -> int:
    try:
        nltk_data.download(args.target_directory)
    except nltk_data.MissingNltkDataError as error:
        print(error)
        return 1
    print(f"NLTK data ready: {', '.join(nltk_data.REQUIRED_RESOURCES.values())}")
    return 0


def _set_role(args: argparse.Namespace) -> int:
    return _update_user(args.email, f"is now {args.role.value}", set_role, args.role)


def _set_plan(args: argparse.Namespace) -> int:
    return _update_user(args.email, f"is now on the {args.plan.value} plan", set_plan, args.plan)


def _update_user(email: str, outcome: str, change: Any, value: Any) -> int:
    async def run() -> bool:
        engine = create_engine(BackendSettings().database_url)
        try:
            found: bool = await change(create_session_factory(engine), email, value)
            return found
        finally:
            await engine.dispose()

    if not asyncio.run(run()):
        print(f"No user with email {email}.")
        return 1
    print(f"{email} {outcome}.")
    return 0


def _worker(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO)
    create_worker(BackendSettings()).run()
    return 0

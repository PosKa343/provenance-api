import json
import os
import subprocess
from pathlib import Path

import anyio
import httpx
import joblib
import nltk
import pytest

from provenance_api.cli import main
from provenance_api.ml.train import METADATA_PATH
from tests.conftest import Headers

DATASET = Path(__file__).parent.parent / "data" / "ai_detection_dataset.csv"


def test_train_writes_a_model_and_its_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["train", str(DATASET), "--output-dir", str(tmp_path)])

    assert exit_code == 0
    written = json.loads((tmp_path / "model.json").read_text())
    committed = json.loads(METADATA_PATH.read_text())
    assert written["dataset"] == committed["dataset"]
    assert written["feature_names"] == committed["feature_names"]
    assert joblib.load(tmp_path / "model.joblib").coef_.shape == (1, 29)
    assert written["model_version"] in capsys.readouterr().out


def test_train_with_a_missing_dataset_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["train", str(tmp_path / "nope.csv"), "--output-dir", str(tmp_path)])

    assert exit_code == 1
    assert "Dataset not found" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_nltk_data_downloads_every_required_package_to_the_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requested: list[tuple[str, str | None]] = []

    def fake_download(package: str, download_dir: str | None = None, quiet: bool = False) -> bool:
        requested.append((package, download_dir))
        return True

    monkeypatch.setattr(nltk, "download", fake_download)

    exit_code = main(["nltk-data", str(tmp_path)])

    assert exit_code == 0
    assert requested == [
        ("punkt_tab", str(tmp_path)),
        ("averaged_perceptron_tagger_eng", str(tmp_path)),
        ("cmudict", str(tmp_path)),
    ]


def test_nltk_data_reports_a_failed_download(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(nltk, "download", lambda *args, **kwargs: False)

    exit_code = main(["nltk-data"])

    assert exit_code == 1
    assert "Could not download NLTK package punkt_tab" in capsys.readouterr().out


def test_set_role_rejects_an_unknown_role(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["set-role", "someone@example.com", "superuser"])

    assert exit_info.value.code == 2
    assert "invalid Role value" in capsys.readouterr().err


def run_installed_command(*args: str, database_url: str) -> subprocess.CompletedProcess[str]:
    """Run the real `provenance-api` script, the way a person or the Dockerfile does."""
    return subprocess.run(  # noqa: S603 - fixed command, test-controlled arguments
        ["provenance-api", *args],  # noqa: S607
        env={**os.environ, "PROVENANCE_DATABASE_URL": database_url},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.anyio
async def test_installed_command_promotes_a_user(
    client: httpx.AsyncClient, member: Headers, database_url: str
) -> None:
    result = await anyio.to_thread.run_sync(
        lambda: run_installed_command(
            "set-role", "member@example.com", "admin", database_url=database_url
        )
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "member@example.com is now admin."
    assert (await client.get("/v1/me", headers=member)).json()["role"] == "admin"


@pytest.mark.anyio
async def test_installed_command_reports_an_unknown_email(
    client: httpx.AsyncClient, member: Headers, database_url: str
) -> None:
    result = await anyio.to_thread.run_sync(
        lambda: run_installed_command(
            "set-role", "nobody@example.com", "admin", database_url=database_url
        )
    )

    assert result.returncode == 1
    assert result.stdout.strip() == "No user with email nobody@example.com."


@pytest.mark.anyio
async def test_installed_command_changes_a_plan(
    client: httpx.AsyncClient, member: Headers, database_url: str
) -> None:
    result = await anyio.to_thread.run_sync(
        lambda: run_installed_command(
            "set-plan", "member@example.com", "pro", database_url=database_url
        )
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "member@example.com is now on the pro plan."
    assert (await client.get("/v1/me", headers=member)).json()["plan"] == "pro"

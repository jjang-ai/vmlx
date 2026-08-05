from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HYGIENE_SCRIPT = ROOT / "scripts/check-public-repo-hygiene.sh"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )


def _commit_forbidden_path(repo: Path, branch: str) -> None:
    _git(repo, "switch", "-c", branch)
    notes = repo / "notes"
    notes.mkdir()
    (notes / "private.md").write_text("private evidence\n", encoding="utf-8")
    _git(repo, "add", "notes/private.md")
    _git(repo, "commit", "-m", f"add {branch} private artifact")


def _public_repo(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "--bare", str(remote)],
        check=True,
        text=True,
        capture_output=True,
    )
    repo = tmp_path / "work"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Public Hygiene Test")
    _git(repo, "config", "user.email", "hygiene@example.com")
    (repo / "README.md").write_text("# Public repository\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial public commit")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", "main")
    return repo, remote


def _run_hygiene(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(HYGIENE_SCRIPT)],
        cwd=repo,
        check=False,
        text=True,
        capture_output=True,
    )


def _assert_temporary_public_refs_removed(repo: Path) -> None:
    refs = _git(
        repo,
        "for-each-ref",
        "--format=%(refname)",
        "refs/vmlx-public-hygiene",
    ).stdout
    assert refs == ""


def test_local_only_private_branch_and_tag_do_not_poison_public_history(tmp_path: Path):
    repo, _ = _public_repo(tmp_path)
    _commit_forbidden_path(repo, "local-private")
    _git(repo, "tag", "local-private-tag")
    _git(repo, "switch", "main")

    result = _run_hygiene(repo)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "Public repository hygiene check passed.\n"
    _assert_temporary_public_refs_removed(repo)


def test_forbidden_artifact_on_public_branch_fails(tmp_path: Path):
    repo, _ = _public_repo(tmp_path)
    _commit_forbidden_path(repo, "public-private")
    _git(repo, "push", "origin", "HEAD:refs/heads/public-private")
    _git(repo, "switch", "main")

    result = _run_hygiene(repo)

    assert result.returncode == 1
    assert "public repository history contains forbidden" in result.stderr
    assert "notes/private.md" in result.stderr
    _assert_temporary_public_refs_removed(repo)


def test_forbidden_artifact_on_public_tag_fails(tmp_path: Path):
    repo, _ = _public_repo(tmp_path)
    _commit_forbidden_path(repo, "tag-private")
    _git(repo, "tag", "public-private-tag")
    _git(repo, "push", "origin", "refs/tags/public-private-tag")
    _git(repo, "switch", "main")

    result = _run_hygiene(repo)

    assert result.returncode == 1
    assert "public repository history contains forbidden" in result.stderr
    assert "notes/private.md" in result.stderr
    _assert_temporary_public_refs_removed(repo)


def test_published_dsv4_contract_docs_are_exactly_allowlisted(tmp_path: Path):
    repo, _ = _public_repo(tmp_path)
    docs = repo / "docs" / "development"
    docs.mkdir(parents=True)
    (docs / "dsv4-encoder-contract.md").write_text(
        "# Public DSV4 encoder contract\n",
        encoding="utf-8",
    )
    (docs / "dsv4-decode-acceptance.md").write_text(
        "# Public DSV4 decode acceptance\n",
        encoding="utf-8",
    )
    _git(repo, "add", "docs/development/dsv4-encoder-contract.md")
    _git(repo, "add", "docs/development/dsv4-decode-acceptance.md")
    _git(repo, "commit", "-m", "publish DSV4 contract docs")
    _git(repo, "push", "origin", "main")

    result = _run_hygiene(repo)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "Public repository hygiene check passed.\n"
    _assert_temporary_public_refs_removed(repo)

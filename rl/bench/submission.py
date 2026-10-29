"""Leaderboard submissions: a policy class plus the parameters to build it.

Each submission lives in ``submissions/<id>/submission.json``::

    {
      "name": "PPO",
      "kind": "rl",
      "policy": "rl.bench.policies.sb3:SB3Policy",
      "params": {"algo": "ppo", "timesteps": 100000},
      "trials": 4,
      "authors": ["..."],
      "url": "https://...",
      "description": "..."
    }

``policy`` is ``package.module:Class`` for installed code, or
``policy.py:Class`` for a file shipped in the submission directory.
``trials`` counts the configurations evaluated on validation data before
this one was chosen, including itself; the leaderboard deflates Sharpe
ratios by it. Frozen live-track state is stored under
``submissions/<id>/frozen/<task>/``. ``ledger`` (optional) lists the task ids
on which the submission commits a decision before every session open
(:mod:`rl.bench.ledger`); entries are stored under
``submissions/<id>/ledger/<task>/``.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rl.bench.policy import Policy

KINDS = ("baseline", "rl", "forecast", "llm", "other")
SUBMISSIONS_DIR = Path("submissions")
_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


@dataclass(frozen=True)
class Submission:
    id: str
    name: str
    kind: str
    policy: str
    params: dict = field(default_factory=dict)
    trials: int = 1
    authors: tuple[str, ...] = ()
    url: str = ""
    description: str = ""
    ledger: tuple[str, ...] = ()
    directory: Path | None = None

    def __post_init__(self):
        if not _ID.match(self.id):
            raise ValueError("Submission id {!r} must be lowercase letters, digits and dashes".format(self.id))
        if self.kind not in KINDS:
            raise ValueError("Submission kind must be one of {}".format(KINDS))
        if ":" not in self.policy:
            raise ValueError("policy must be 'module:Class'")
        if not isinstance(self.trials, int) or self.trials < 1:
            raise ValueError("trials must be a positive integer")
        json.dumps(self.params)  # parameters must round-trip through JSON
        if isinstance(self.ledger, str) or not all(isinstance(t, str) and t for t in self.ledger):
            raise ValueError("ledger must be a list of task ids")
        if len(set(self.ledger)) != len(self.ledger):
            raise ValueError("ledger task ids must be unique")

    @classmethod
    def load(cls, directory) -> Submission:
        directory = Path(directory)
        spec = json.loads((directory / "submission.json").read_text())
        allowed = {"name", "kind", "policy", "params", "trials", "authors", "url", "description", "ledger"}
        unknown = set(spec) - allowed
        if unknown:
            raise ValueError("Unknown submission fields {} in {}".format(sorted(unknown), directory))
        spec["authors"] = tuple(spec.get("authors", ()))
        spec["ledger"] = tuple(spec.get("ledger", ()))
        return cls(id=directory.name, directory=directory, **spec)

    def policy_class(self) -> type[Policy]:
        """The policy class: ``package.module:Class``, or ``file.py:Class``
        relative to the submission directory for code shipped with it."""
        module_name, _, class_name = self.policy.partition(":")
        if module_name.endswith(".py"):
            if self.directory is None:
                raise ValueError("Submission {} has no directory to load {} from".format(self.id, module_name))
            path = self.directory / module_name
            spec = importlib.util.spec_from_file_location(
                "submissions.{}.{}".format(self.id.replace("-", "_"), path.stem), path
            )
            if spec is None or not path.is_file():
                raise FileNotFoundError("Policy file {} not found".format(path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_name)
        policy_class = getattr(module, class_name)
        if not (isinstance(policy_class, type) and issubclass(policy_class, Policy)):
            raise TypeError("{} is not a Policy subclass".format(self.policy))
        return policy_class

    def frozen_dir(self, task_id: str) -> Path:
        if self.directory is None:
            raise ValueError("Submission {} has no directory".format(self.id))
        return self.directory / "frozen" / task_id

    def ledger_dir(self, task_id: str) -> Path:
        if self.directory is None:
            raise ValueError("Submission {} has no directory".format(self.id))
        return self.directory / "ledger" / task_id

    def fingerprint(self) -> str:
        """SHA-256 of everything that shapes this submission's historical record.

        It covers the fields a record stores (``name``, ``kind``, ``policy``,
        ``params``, ``trials``) and every file shipped in the directory except
        ``submission.json``, the CI-written ``frozen/`` and ``ledger/``,
        ``__pycache__`` and dotfiles. Editing ``description``, ``authors`` or
        ``url`` keeps a record current; any other change makes it stale.
        """
        spec = {"name": self.name, "kind": self.kind, "policy": self.policy, "params": self.params, "trials": self.trials}
        digest = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode())
        if self.directory is None:
            return digest.hexdigest()
        for path in sorted(self.directory.rglob("*"), key=lambda p: p.relative_to(self.directory).as_posix()):
            parts = path.relative_to(self.directory).parts
            if (
                not path.is_file()
                or parts == ("submission.json",)
                or parts[0] in ("frozen", "ledger")
                or any(part == "__pycache__" or part.startswith(".") for part in parts)
            ):
                continue
            digest.update(b"\0" + "/".join(parts).encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()


def discover(root=SUBMISSIONS_DIR) -> dict[str, Submission]:
    """Every submission under ``root``, keyed by id."""
    root = Path(root)
    found = {}
    for spec in sorted(root.glob("*/submission.json")):
        submission = Submission.load(spec.parent)
        found[submission.id] = submission
    return found

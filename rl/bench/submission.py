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

Add ``"retired": "YYYY-MM-DD"`` to stop a submission after that date: it is
skipped by ``run-all``, ``freeze``, ``live`` and the ledger, leaves the ranking
and is listed under "Retired" with its records intact.

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

import ast
import datetime as dt
import hashlib
import importlib
import importlib.util
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

from rl.bench.info import validate_sources
from rl.bench.policy import Policy

KINDS = ("baseline", "rl", "forecast", "llm", "other")
SUBMISSIONS_DIR = Path("submissions")
_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


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
    retired: str | None = None
    directory: Path | None = None
    info_sources: tuple[dict, ...] = ()
    model_cutoffs: tuple[dict, ...] = ()
    contamination: str | None = None

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
        object.__setattr__(self, "info_sources", validate_sources(self.info_sources))
        if self.contamination not in (None, "possible"):
            raise ValueError('contamination must be "possible" or omitted')
        if not isinstance(self.model_cutoffs, (list, tuple)):
            raise ValueError("model_cutoffs must be a list")
        models = set()
        for entry in self.model_cutoffs:
            if not isinstance(entry, dict) or set(entry) != {"model", "training_cutoff", "source_url"}:
                raise ValueError("model_cutoffs need model, training_cutoff and source_url")
            if not isinstance(entry["model"], str) or not entry["model"].strip() or entry["model"] in models:
                raise ValueError("model_cutoffs need unique nonempty model ids")
            models.add(entry["model"])
            cutoff = entry["training_cutoff"]
            if not isinstance(cutoff, str) or date.fromisoformat(cutoff).isoformat() != cutoff:
                raise ValueError("training_cutoff must be YYYY-MM-DD")
            url = urlparse(entry["source_url"])
            if url.scheme not in ("https", "http") or not url.netloc:
                raise ValueError("model cutoff source_url must be an HTTP(S) URL")
        object.__setattr__(self, "model_cutoffs", tuple(dict(entry) for entry in self.model_cutoffs))
        if self.retired is not None and not _is_date(self.retired):
            raise ValueError("retired must be a YYYY-MM-DD date")

    def is_retired(self, today: dt.date | None = None) -> bool:
        """Whether ``today`` (default: the current UTC date) is after the ``retired`` date."""
        today = today or dt.datetime.now(dt.timezone.utc).date()
        return self.retired is not None and today > dt.date.fromisoformat(self.retired)

    @classmethod
    def load(cls, directory) -> Submission:
        directory = Path(directory)
        spec = json.loads((directory / "submission.json").read_text())
        allowed = {
            "name", "kind", "policy", "params", "trials", "authors", "url", "description", "ledger",
            "info_sources", "model_cutoffs", "contamination", "retired",
        }
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
        # Preserve fingerprints (and record bytes) for existing declarations.
        for name in ("info_sources", "model_cutoffs", "contamination"):
            if getattr(self, name):
                spec[name] = getattr(self, name)
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

    def code_sources(self) -> dict[str, bytes]:
        """Source files that define this submission's policy, by path.

        The policy's module (a file shipped in the submission directory, or an
        importable module) and every ``rl.bench`` module and package it imports
        directly or transitively. Imports are read from the source, so the
        result does not depend on what happens to be loaded; third-party
        packages are identified by the versions in the record instead.
        """
        module_name = self.policy.partition(":")[0]
        sources: dict[str, bytes] = {}
        pending: list[str] = []
        if module_name.endswith(".py"):
            if self.directory is None:
                raise ValueError("Submission {} has no directory to load {} from".format(self.id, module_name))
            source = (self.directory / module_name).read_bytes()
            sources["submissions/{}/{}".format(self.id, module_name)] = source
            pending.extend(_bench_imports(source, None))
        else:
            pending.append(module_name)
        while pending:
            name = pending.pop()
            try:
                spec = importlib.util.find_spec(name)
            except ModuleNotFoundError:  # "from module import name" where name is not a submodule
                continue
            if spec is None or not (spec.origin or "").endswith(".py"):
                continue
            label = _module_label(name, spec)
            if label in sources:
                continue
            source = Path(spec.origin).read_bytes()
            sources[label] = source
            pending.extend(_bench_imports(source, name if spec.submodule_search_locations else name.rpartition(".")[0]))
            if name.startswith("rl.bench."):  # importing a module runs its parent packages
                pending.append(name.rpartition(".")[0])
        return sources

    def code_sha256(self) -> str:
        """SHA-256 over the sorted (path, bytes) of :meth:`code_sources`."""
        return _digest(self.code_sources())

    def policy_sha256(self) -> str:
        """SHA-256 of the policy's own module alone, without the ``rl.bench`` code it imports.

        It changes only when the policy's source does, not when the harness around it is edited.
        """
        module_name = self.policy.partition(":")[0]
        if module_name.endswith(".py"):
            if self.directory is None:
                raise ValueError("Submission {} has no directory to load {} from".format(self.id, module_name))
            return _digest({"submissions/{}/{}".format(self.id, module_name): (self.directory / module_name).read_bytes()})
        spec = importlib.util.find_spec(module_name)
        if spec is None or not (spec.origin or "").endswith(".py"):
            raise ModuleNotFoundError("Policy module {} has no Python source".format(module_name))
        return _digest({_module_label(module_name, spec): Path(spec.origin).read_bytes()})


def discover(root=SUBMISSIONS_DIR) -> dict[str, Submission]:
    """Every submission under ``root``, keyed by id."""
    root = Path(root)
    found = {}
    for spec in sorted(root.glob("*/submission.json")):
        submission = Submission.load(spec.parent)
        found[submission.id] = submission
    return found


def _module_label(name: str, spec) -> str:
    return name.replace(".", "/") + ("/__init__.py" if spec.submodule_search_locations else ".py")


def _digest(sources: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for label, source in sorted(sources.items()):
        digest.update(label.encode() + b"\0" + len(source).to_bytes(8, "big") + source)
    return digest.hexdigest()


def _bench_imports(source: bytes, package: str | None) -> set[str]:
    """``rl.bench`` module names that ``source`` imports; ``package`` resolves relative imports."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level and package is None:
                continue
            base = importlib.util.resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module
            names.add(base)
            names.update("{}.{}".format(base, alias.name) for alias in node.names)
    return {name for name in names if name == "rl.bench" or name.startswith("rl.bench.")}


def _is_date(value) -> bool:
    try:
        return isinstance(value, str) and bool(_DATE.match(value)) and dt.date.fromisoformat(value) is not None
    except ValueError:
        return False

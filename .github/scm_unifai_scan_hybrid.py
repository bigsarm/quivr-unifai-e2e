#!/usr/bin/env python3
"""Lineaje UnifAI Policy Scanner â€” GitHub Actions edition.

Scans already-checked-out source code against Lineaje AI security policies and
optionally opens a remediation PR built from the LLM remediation ``fix_code`` patches
the policy engine returns (guardrail stubs are not inserted); the PR commits the
patched files. The step summary also prints the exact ``entities.aientity.json`` and
``findings.aifinding.json`` the server uploaded to Lineaje â€” in hybrid mode those are
the only scan artifacts that leave the MCP host. Designed to run on a GitHub-managed
(or self-hosted) runner where the repository is already checked out.

Self-contained: the only SCM code here is a minimal GitHub REST client
(:class:`GitHubClient`, defined below) covering the branch/commit/PR calls the
remediation step makes, so this script is the only file a workflow needs to
copy. Nothing outside the Python stdlib is required beyond the ``mcp`` SDK.

Every file under ``--source-path`` is scanned. There is no exclusion list and no
special handling for dependency manifests: the walk collects everything it
finds, splits it into batches of ``UNIFAI_FILE_BATCH_SIZE`` files, and uploads
each batch archive as-is. Trimming the scan input (e.g. dropping ``.git``) is
the caller's job â€” the workflow does it before invoking this script.

Usage::

    python scm_unifai_scan.py --source-path . --create-fix-pr

Output:

* **stdout** â€” the markdown policy report. The workflow redirects this into
  ``$GITHUB_STEP_SUMMARY`` so it renders on the run summary page rather than
  filling the step log.
* **stderr** â€” progress logs, ending with a one-line result such as
  ``Result: âŒ Not Compliant â€” 2 violation(s)``.

Required environment variable::

    LINEAJE_PAT_TOKEN  â€” Lineaje refresh token (exchanged for short-lived access tokens)

Optional environment variables::

    GITHUB_TOKEN / GH_TOKEN  â€” needed by --create-fix-pr to push the branch and open the PR
    UNIFAI_FILE_BATCH_SIZE   â€” files per batch (default 100; 0 puts everything in one batch)
    MCP_SERVER_URL           â€” override the Lineaje MCP endpoint

Exit codes::

    0 â€” scan completed (see the report for compliance status)
    1 â€” runtime error (every batch failed, or an unhandled exception)
    2 â€” configuration error (auth failure, missing repo/branch)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import fnmatch
import json
import ast
import logging
import os
import pathlib
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("gha_repo_scan")

# ===========================================================================
# Constants
# ===========================================================================

MCP_SERVER_URL = "https://172.206.26.109/mcp"  # Put in your VM IP Address here


def _mcp_http_client_with_extra_ca(headers=None, timeout=None, auth=None):
    """Same defaults as mcp's create_mcp_http_client, plus (if configured) one
    extra CA added to â€” not replacing â€” the normal system/certifi trust store.

    The Wipro VM's Caddy `tls internal` cert is self-signed, and httpx (which
    the mcp SDK's streamablehttp_client uses) defaults to certifi's public CA
    bundle only â€” unlike stdlib ssl/requests, it does NOT read
    SSL_CERT_FILE/REQUESTS_CA_BUNDLE on its own. A fresh CI runner with no
    extra trust config therefore fails the TLS handshake before any request is
    even sent, surfacing to a caller only as a generic "unhandled errors in a
    TaskGroup (1 sub-exception)", with nothing logged server-side because the
    connection never completes.

    No certificate content lives in this script: point MCP_CA_BUNDLE (or the
    same REQUESTS_CA_BUNDLE / SSL_CERT_FILE convention scripts/run_demo_scan.sh
    already uses) at a CA file supplied separately by the workflow â€” e.g. a
    repo/org secret written to a temp file in an earlier step. Unset, this is
    a no-op: behavior is identical to the SDK's own create_mcp_http_client.
    """
    import ssl

    import httpx

    # Defaults to "1" (insecure) when unset â€” set MCP_TLS_INSECURE_SKIP_VERIFY=0/false/no
    # explicitly to turn verification back on. Only leave this default on against a
    # known, private endpoint reachable solely by this workflow.
    insecure = (os.environ.get("MCP_TLS_INSECURE_SKIP_VERIFY", "1") or "1").strip().lower() in ("1", "true", "yes")
    verify: Any
    if insecure:
        logger.warning(
            "MCP_TLS_INSECURE_SKIP_VERIFY is set â€” TLS certificate and hostname "
            "verification are DISABLED for the MCP connection. Do not use this "
            "against anything but a known, private endpoint."
        )
        verify = False
    else:
        ctx = ssl.create_default_context()
        ca_bundle_path = (
            os.environ.get("MCP_CA_BUNDLE")
            or os.environ.get("REQUESTS_CA_BUNDLE")
            or os.environ.get("SSL_CERT_FILE")
            or ""
        ).strip()
        if ca_bundle_path:
            ctx.load_verify_locations(cafile=ca_bundle_path)
        verify = ctx

    kwargs: Dict[str, Any] = {"follow_redirects": True, "verify": verify}
    kwargs["timeout"] = timeout if timeout is not None else httpx.Timeout(30, read=300)
    if headers is not None:
        kwargs["headers"] = headers
    if auth is not None:
        kwargs["auth"] = auth
    return httpx.AsyncClient(**kwargs)



from contextlib import asynccontextmanager as _asynccontextmanager
from datetime import timedelta as _timedelta


@_asynccontextmanager
async def _streamablehttp_client_compat(
    url,
    headers=None,
    timeout=30,
    sse_read_timeout=60 * 5,
    terminate_on_close=True,
    httpx_client_factory=None,
    auth=None,
):
    import httpx
    from mcp.client.streamable_http import streamable_http_client

    factory = httpx_client_factory or _mcp_http_client_with_extra_ca
    timeout_seconds = timeout.total_seconds() if isinstance(timeout, _timedelta) else timeout
    sse_read_timeout_seconds = (
        sse_read_timeout.total_seconds() if isinstance(sse_read_timeout, _timedelta) else sse_read_timeout
    )
    client = factory(
        headers=headers,
        timeout=httpx.Timeout(timeout_seconds, read=sse_read_timeout_seconds),
        auth=auth,
    )
    async with client:
        async with streamable_http_client(
            url, http_client=client, terminate_on_close=terminate_on_close,
        ) as streams:
            if len(streams) == 2:
                streams = (*streams, None)
            yield streams


MAX_SCAN_WORKERS = 4
REMEDIATION_BRANCH_PREFIX = "remediation/unifai-gha"
DEFAULT_UNIFAI_FILE_BATCH_SIZE = 100
GITHUB_PR_BODY_SAFE_LIMIT = 60000
# Azure DevOps rejects PR descriptions longer than 4000 characters.
AZURE_PR_DESCRIPTION_LIMIT = 4000

# Kept inline because this script is copied to customer CI runners and must
# remain standalone. These values mirror scripts/gha_repo_scan.py.
EVIDENCE_TYPE_SCM_SCAN = "scm_scan"
MANIFEST_FILE_PATTERNS = frozenset({
    "requirements.txt", "Pipfile", "pyproject.toml", "setup.py", "setup.cfg",
    "*.toml", "environment.yml", "environment.yaml", "poetry.lock", "Pipfile.lock",
    "package.json", "yarn.lock", "pom.xml", "project.xml", "build.gradle",
    "build.gradle.kts", "build.gradle.mustache", "build.sbt", "Gemfile", "go.mod",
    "Cargo.toml", "packages.config", "*.csproj", "*.fsproj", "*.vbproj",
    "nuget.config", "Directory.Packages.props", "*.sln", "*.slnx", "vcpkg.json",
    ".vcpkg-root", "composer.json", "Package.swift", "pubspec.yaml", "mix.exs",
    "*.gemspec", "config.json",
})
LOCKFILE_BASENAMES = frozenset({
    "bun.lockb", "flake.lock", "package-lock.json", "pnpm-lock.yaml", "bun.lock",
    "gradle.lockfile", "Gemfile.lock", "go.sum", "Cargo.lock", "composer.lock",
    "Package.resolved", "pubspec.lock", "mix.lock", "packages.lock.json",
})
_MANIFEST_LOCKFILE_EXCEPTIONS = frozenset({"yarn.lock", "poetry.lock", "Pipfile.lock"})
ARCHIVE_EXCLUDE_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", "venv", ".venv",
    "env", ".tox", "htmlcov", ".mypy_cache", ".ruff_cache", "node_modules",
    ".yarn", ".pnp", "dist", "build", ".next", ".nuxt", "out", "coverage",
    ".cache", "target", ".gradle", ".m2", "Pods", ".expo", ".idea", ".vscode",
    ".lineaje-aiepo-security", ".lineaje", "migrations", "alembic",
})
ARCHIVE_EXCLUDE_GLOBS = frozenset({
    "*.secret", "*.key", "*.pem", "*.env.*", "*.zip", "*.tar", "*.tar.gz",
    "*.jar", "*.war", "*.swp", "*.swo", "*.lock", "package-lock.json",
    "Gemfile.lock", "Cargo.lock", "composer.lock", "*.min.js", "*.min.css", "*.map",
    "*_pb2.py", "*.pb.go", "*.pb.cc", "*.pb.h", "*.snap", "*README*",
    "*readme*", "*Dockerfile*", "*dockerfile*", "*.dockerfile", "*.dockerignore",
})
BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".webp", ".svg", ".woff",
    ".woff2", ".ttf", ".eot", ".otf", ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".ppt", ".pptx", ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".exe",
    ".dll", ".so", ".dylib", ".class", ".jar", ".war", ".pyc", ".pyo", ".o",
    ".a", ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac", ".db", ".sqlite",
    ".sqlite3",
})
_ARCHIVE_EXCLUDE_DIR_GLOBS = (".venv-*", "venv-*")


def is_lockfile_basename(name: str) -> bool:
    base = os.path.basename(str(name or ""))
    lower = base.lower()
    exceptions = {n.lower() for n in _MANIFEST_LOCKFILE_EXCEPTIONS}
    lockfiles = {n.lower() for n in LOCKFILE_BASENAMES}
    if lower in exceptions:
        return False
    return (
        lower in lockfiles or lower.endswith(".lock") or lower.endswith(".lockfile")
        or lower.endswith("-lock.json") or lower.endswith("-lock.yaml")
    )


def _is_manifest_basename(name: str) -> bool:
    if is_lockfile_basename(name):
        return False
    return name in MANIFEST_FILE_PATTERNS or any(
        fnmatch.fnmatch(name, pattern)
        for pattern in MANIFEST_FILE_PATTERNS if "*" in pattern or "?" in pattern
    )


def archive_exclude_globs_keeping_manifests() -> Tuple[str, ...]:
    return tuple(sorted(pattern for pattern in ARCHIVE_EXCLUDE_GLOBS if pattern not in MANIFEST_FILE_PATTERNS))


def pin_manifests_to_first_batch(
    file_list: List[str], batch_size: int,
) -> Tuple[List[List[str]], List[str], List[str]]:
    if batch_size <= 0:
        batch_size = max(1, len(file_list) or 1)
    filtered = [path for path in file_list if not is_lockfile_basename(os.path.basename(path))]
    manifests = [path for path in filtered if _is_manifest_basename(os.path.basename(path))]
    code = [path for path in filtered if not _is_manifest_basename(os.path.basename(path))]
    code_batches = [code[i:i + batch_size] for i in range(0, len(code), batch_size)]
    if not code_batches:
        return ([manifests] if manifests else []), code, manifests
    return [manifests + code_batches[0], *code_batches[1:]], code, manifests


def split_batch_keeping_manifests_whole(
    batch_files: List[str],
) -> Optional[Tuple[List[str], List[str]]]:
    if len(batch_files) < 2:
        return None
    manifests = [path for path in batch_files if _is_manifest_basename(os.path.basename(path))]
    code = [path for path in batch_files if not _is_manifest_basename(os.path.basename(path))]
    if not code:
        return None
    if len(code) == 1:
        return (manifests, code) if manifests else None
    midpoint = len(code) // 2
    return manifests + code[:midpoint], code[midpoint:]

_DEFAULT_LINEAJE_TOKEN_REFRESH_SKEW_SEC = 120
_LINEAJE_NATIVE_RENEW_ACCESS_TOKEN_URL_PROD = (
    "https://lineaje-identity-service.v2.prod.veedna.com"
    "/lineajeidentity/api/v1/auth/native/renew-access-token"
)


# ===========================================================================
# Azure DevOps client â€” remediation branch, commit, pull request
#
# Only the calls _create_fix_pr makes, over urllib.request from the stdlib.
# ===========================================================================

_NULL_OBJECT_ID = "0" * 40


class AzureDevOpsClient:
    """Minimal Azure DevOps Git REST client for opening remediation PRs.

    Every URL is built from the collection URI the agent already exposes as
    ``$(System.TeamFoundationCollectionUri)``, so the same client works against
    Azure DevOps Services (``https://dev.azure.com/{org}/``) and against an
    on-premises Azure DevOps Server collection.

    Both token flavours a pipeline can supply are accepted and auto-detected:
    ``$(System.AccessToken)`` is a JWT and goes out as ``Authorization: Bearer``,
    while a personal access token goes out as HTTP Basic with an empty username,
    which is the only form Azure DevOps accepts for PATs.
    """

    def __init__(self, token: str, org_url: str, project: str, repository: str) -> None:
        self.token = token
        self.org_url = org_url.rstrip("/")
        self.project = project
        self.repository = repository

    # -- request plumbing --------------------------------------------------

    @property
    def _repo_base(self) -> str:
        return (
            f"{self.org_url}/{urllib.parse.quote(self.project, safe='')}"
            f"/_apis/git/repositories/{urllib.parse.quote(self.repository, safe='')}"
        )

    def _auth_header(self) -> str:
        if self.token.count(".") == 2 and self.token.startswith("ey"):
            return f"Bearer {self.token}"
        return "Basic " + base64.b64encode(f":{self.token}".encode()).decode()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": self._auth_header(),
            "Accept": "application/json",
            "User-Agent": "UniFAI-PR-Scanner/1.0",
        }

    def _url(self, endpoint: str, **params: str) -> str:
        query: Dict[str, str] = {"api-version": AZURE_DEVOPS_API_VERSION}
        query.update({k: v for k, v in params.items() if v})
        return f"{self._repo_base}{endpoint}?{urllib.parse.urlencode(query)}"

    def _request(
        self,
        method: str,
        url: str,
        body: Any = None,
        *,
        expected_errors: Optional[set] = None,
    ) -> Any:
        """Issue an HTTP request and return the decoded JSON.

        *expected_errors* is an optional set of HTTP status codes (e.g. ``{404}``)
        that the caller expects and will handle â€” these are logged at DEBUG instead
        of ERROR so they don't pollute output during normal operation.
        """
        headers = self._headers()

        data = json.dumps(body).encode() if body is not None else None
        if data:
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        logger.debug("%s %s", method, url)

        try:
            with urllib.request.urlopen(req) as resp:
                resp_bytes = resp.read()
                if not resp_bytes:
                    return None
                content_type = resp.headers.get("Content-Type", "")
                if "json" not in content_type.lower():
                    # Azure DevOps answers an unauthenticated API call with the
                    # HTML sign-in page and HTTP 200 rather than a 401, so a
                    # non-JSON body here almost always means a rejected token.
                    raise RuntimeError(
                        f"Azure DevOps returned {content_type or 'a non-JSON body'} for {method} {url} â€” "
                        "the token was most likely rejected. Check that the step maps "
                        "SYSTEM_ACCESSTOKEN, or that the PAT has the Code (Read & Write) scope."
                    )
                return json.loads(resp_bytes)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")[:500]
            if expected_errors and exc.code in expected_errors:
                logger.debug("Azure DevOps API %s %s â†’ %s (expected): %s", method, url, exc.code, error_body)
            else:
                logger.error("Azure DevOps API %s %s â†’ %s: %s", method, url, exc.code, error_body)
            raise

    # -- refs --------------------------------------------------------------

    def get_branch_sha(self, branch: str) -> Optional[str]:
        """Tip commit id of *branch*, or None when the branch does not exist."""
        data = self._request("GET", self._url("/refs", filter=f"heads/{branch}"))
        for ref in (data or {}).get("value", []):
            if ref.get("name") == f"refs/heads/{branch}" and ref.get("objectId"):
                return str(ref["objectId"])
        return None

    def create_branch(self, branch_name: str, from_sha: str) -> None:
        """Create ``refs/heads/<branch_name>`` pointing at *from_sha*."""
        resp = self._request("POST", self._url("/refs"), [{
            "name": f"refs/heads/{branch_name}",
            "oldObjectId": _NULL_OBJECT_ID,
            "newObjectId": from_sha,
        }])
        # This endpoint reports per-ref failures inside a 200 response body.
        for result in (resp or {}).get("value", []):
            if not result.get("success", True):
                detail = (
                    result.get("customMessage")
                    or result.get("updateStatus")
                    or "unknown error"
                )
                raise RuntimeError(f"Could not create refs/heads/{branch_name}: {detail}")

    # -- items -------------------------------------------------------------

    def file_exists(self, path: str, ref_sha: str) -> bool:
        """Whether *path* exists at *ref_sha*, which picks ``edit`` vs ``add``."""
        url = self._url(
            "/items",
            **{
                "path": f"/{path.lstrip('/')}",
                "versionDescriptor.version": ref_sha,
                "versionDescriptor.versionType": "commit",
                "includeContent": "false",
                "$format": "json",
            },
        )
        try:
            self._request("GET", url, expected_errors={404})
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            raise

    # -- pushes ------------------------------------------------------------

    def push_commit(
        self,
        branch: str,
        base_sha: str,
        files: Dict[str, bytes],
        message: str,
        *,
        existing: Optional[Dict[str, bool]] = None,
    ) -> str:
        """Push every entry of *files* to *branch* as a single commit on *base_sha*.

        Azure DevOps commits through the Pushes API rather than one call per file
        the way GitHub's Contents API does, so the whole remediation lands as one
        commit and there is no partially-committed branch to clean up on failure.
        """
        existing = existing or {}
        changes: List[Dict[str, Any]] = []
        for path, content in sorted(files.items()):
            rel = path.lstrip("/")
            changes.append({
                "changeType": "edit" if existing.get(rel, True) else "add",
                "item": {"path": f"/{rel}"},
                "newContent": {
                    "content": base64.b64encode(content).decode(),
                    "contentType": "base64encoded",
                },
            })
        resp = self._request("POST", self._url("/pushes"), {
            "refUpdates": [{"name": f"refs/heads/{branch}", "oldObjectId": base_sha}],
            "commits": [{"comment": message, "changes": changes}],
        })
        commits = (resp or {}).get("commits") or [{}]
        return str(commits[0].get("commitId", ""))

    # -- pull requests -----------------------------------------------------

    def create_pull_request(
        self, title: str, source_branch: str, target_branch: str, description: str
    ) -> int:
        resp = self._request("POST", self._url("/pullrequests"), {
            "sourceRefName": f"refs/heads/{source_branch}",
            "targetRefName": f"refs/heads/{target_branch}",
            "title": title[:AZURE_PR_TITLE_LIMIT],
            "description": description[:AZURE_PR_DESCRIPTION_LIMIT],
        })
        return int(resp["pullRequestId"])

    def pull_request_url(self, pr_id: int) -> str:
        return (
            f"{self.org_url}/{urllib.parse.quote(self.project, safe='')}"
            f"/_git/{urllib.parse.quote(self.repository, safe='')}/pullrequest/{pr_id}"
        )

# ===========================================================================
# Token helpers
# ===========================================================================

def _normalize_token(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw).strip().lstrip("ï»¿").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def _normalize_url(url: Optional[str]) -> str:
    if url is None:
        return ""
    u = str(url).strip()
    if len(u) >= 2 and u[0] == u[-1] and u[0] in "\"'":
        u = u[1:-1].strip()
    return u


def _identity_token_response_dict(raw_text: str, *, context: str) -> dict:
    text = raw_text.strip() if raw_text else ""
    try:
        parsed: Any = json.loads(raw_text)
    except json.JSONDecodeError:
        # Some endpoints return a bare JWT string
        parts = text.split(".")
        if context == "renew-access-token" and len(parts) == 3:
            return {"access_token": text}
        raise RuntimeError(f"{context}: response is not valid JSON") from None
    for _ in range(8):
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, str):
            s = parsed.strip()
            if not s:
                raise RuntimeError(f"{context}: empty JSON string where object expected")
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                parts = s.split(".")
                if context == "renew-access-token" and len(parts) == 3:
                    return {"access_token": s}
                raise RuntimeError(f"{context}: server returned error string: {s[:800]}") from None
            continue
        break
    raise RuntimeError(f"{context}: unexpected JSON type after unwrap: {type(parsed).__name__}")


class RefreshTokenTokenManager:
    """Exchange LINEAJE_PAT_TOKEN at the production identity endpoint.

    The hybrid scanner intentionally does not honor endpoint overrides for
    refresh-token exchange. Customer CI configuration may change the MCP
    endpoint, but credentials must always be renewed by the production Lineaje
    identity service.
    """

    def __init__(self, refresh_token: str, renew_access_token_url: Optional[str] = None) -> None:
        self._refresh_token = _normalize_token(refresh_token)
        if not self._refresh_token:
            raise ValueError("LINEAJE_PAT_TOKEN must be non-empty")
        # Keep the optional argument for call-site compatibility, but never let
        # it or LINEAJE_RENEW_ACCESS_TOKEN_URL redirect credentials elsewhere.
        self._renew_url = _LINEAJE_NATIVE_RENEW_ACCESS_TOKEN_URL_PROD
        self._lock = threading.Lock()
        self._access_token = ""
        self._access_deadline = 0.0
        try:
            self._skew_sec = int(os.environ.get(
                "LINEAJE_TOKEN_REFRESH_SKEW_SEC", str(_DEFAULT_LINEAJE_TOKEN_REFRESH_SKEW_SEC)
            ))
        except ValueError:
            self._skew_sec = _DEFAULT_LINEAJE_TOKEN_REFRESH_SKEW_SEC

    def get_access_token(self) -> str:
        with self._lock:
            return self._get_unlocked()

    def _get_unlocked(self) -> str:
        now = time.time()
        if self._access_token and now < self._access_deadline - self._skew_sec:
            return self._access_token
        self._renew()
        if not self._access_token:
            raise RuntimeError("renew-access-token did not return access_token")
        return self._access_token

    def _renew(self) -> None:
        q = urllib.parse.urlencode({"refreshToken": self._refresh_token})
        url = f"{self._renew_url}?{q}"
        req = urllib.request.Request(
            url, data=b"null",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = _identity_token_response_dict(resp.read().decode(), context="renew-access-token")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RuntimeError(f"renew-access-token HTTP {exc.code}: {body[:800]}") from exc
        at = (data.get("access_token") or "").strip()
        if not at:
            raise RuntimeError(f"Token response missing access_token: {data!r}")
        self._access_token = at
        rt = (data.get("refresh_token") or "").strip()
        if rt:
            self._refresh_token = rt
        exp = data.get("expires_in")
        try:
            exp_sec = int(exp) if exp is not None else 3600
        except (TypeError, ValueError):
            exp_sec = 3600
        self._access_deadline = time.time() + max(60, exp_sec)
        logger.debug("Access token renewed; expires in %ds", exp_sec)


def build_bearer_getter() -> Callable[[], str]:
    pat = _normalize_token(os.environ.get("LINEAJE_PAT_TOKEN", ""))
    if not pat:
        raise RuntimeError("LINEAJE_PAT_TOKEN is not set")
    mgr = RefreshTokenTokenManager(pat)
    return mgr.get_access_token

# ===========================================================================
# File collection
# ===========================================================================

def collect_repo_files(local_path: str, exclude: Optional[List[str]] = None) -> List[str]:
    """Collect the same useful source set as the standalone GHA scanner.

    Explicit ``--exclude`` paths remain supported for Azure Pipelines. Common
    VCS/vendor/build directories, generated files, secrets, binaries, and
    dependency lockfiles are filtered without deleting anything in the checkout.
    """
    excluded = {_norm_rel_path(e) for e in (exclude or []) if e and e.strip()}
    exclude_globs = archive_exclude_globs_keeping_manifests()

    def explicitly_excluded(path: str) -> bool:
        normalized = _norm_rel_path(path)
        return any(normalized == item or normalized.startswith(item + "/") for item in excluded)

    def excluded_dir(name: str, rel_path: str) -> bool:
        return (
            explicitly_excluded(rel_path)
            or name in ARCHIVE_EXCLUDE_DIRS
            or any(fnmatch.fnmatch(name, pattern) for pattern in _ARCHIVE_EXCLUDE_DIR_GLOBS)
        )

    file_list: List[str] = []
    for root, dirs, filenames in os.walk(local_path):
        rel_root = _norm_rel_path(os.path.relpath(root, local_path))
        if rel_root == ".":
            rel_root = ""
        dirs[:] = [
            directory for directory in dirs
            if not excluded_dir(directory, f"{rel_root}/{directory}" if rel_root else directory)
        ]
        for fname in filenames:
            rel_path = os.path.relpath(os.path.join(root, fname), local_path).replace("\\", "/")
            if explicitly_excluded(rel_path) or is_lockfile_basename(fname):
                continue
            if pathlib.Path(fname).suffix.lower() in BINARY_EXTENSIONS:
                continue
            if any(fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(fname, pattern)
                   for pattern in exclude_globs):
                continue
            file_list.append(rel_path)
    return sorted(file_list)

# ===========================================================================
# Archive creation
# ===========================================================================

def _norm_archive_rel_path(p: str) -> str:
    s = p.strip().replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    return s


def create_batch_archive(
    source_dir: str,
    archive_dir: str,
    file_subset: List[str],
    source_code_repo: str,
    branch: str,
    head_sha: str,
    batch_index: Any = 0,
    run_id: str = "",
    manifest_files: Optional[List[str]] = None,
) -> str:
    extra_manifests = [path for path in (manifest_files or []) if path not in file_subset]
    archive_files = list(file_subset) + extra_manifests
    archive_path = os.path.join(archive_dir, f"repo_scan_batch_{batch_index}.zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel_path in archive_files:
            full_path = os.path.join(source_dir, rel_path)
            if os.path.isfile(full_path):
                zf.write(full_path, rel_path)
        metadata = {
            "scan_source": "ado_repo_scan",
            "repo": source_code_repo,
            "branch": branch,
            "head_sha": head_sha,
            "scan_type": "full_repository",
            "batch_index": batch_index,
            "batch_file_count": len(file_subset),
            "manifest_file_count": len(extra_manifests),
        }
        zf.writestr("user_metadata.json", json.dumps(metadata, indent=2))
    size_kb = os.path.getsize(archive_path) // 1024
    logger.info(
        "Batch archive #%s: %d files + %d manifests, %d KB",
        batch_index, len(file_subset), len(extra_manifests), size_kb,
    )
    return archive_path


def _batch_size(total_files: int) -> int:
    raw = (os.environ.get("UNIFAI_FILE_BATCH_SIZE") or "").strip()
    if not raw:
        return DEFAULT_UNIFAI_FILE_BATCH_SIZE
    try:
        size = int(raw)
    except ValueError:
        return DEFAULT_UNIFAI_FILE_BATCH_SIZE
    if size <= 0:
        return max(1, total_files)
    return size

# ===========================================================================
# MCP scan (SDK path only)
# ===========================================================================

from contextlib import asynccontextmanager as _asynccontextmanager
from datetime import timedelta as _timedelta


@_asynccontextmanager
async def streamablehttp_client(
    url,
    headers=None,
    timeout=30,
    sse_read_timeout=60 * 5,
    terminate_on_close=True,
    httpx_client_factory=None,
    auth=None,
):
    import httpx
    from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

    timeout_seconds = timeout.total_seconds() if isinstance(timeout, _timedelta) else timeout
    sse_read_timeout_seconds = (
        sse_read_timeout.total_seconds() if isinstance(sse_read_timeout, _timedelta) else sse_read_timeout
    )
    httpx_timeout = httpx.Timeout(timeout_seconds, read=sse_read_timeout_seconds)

    # MCP_CA_BUNDLE / REQUESTS_CA_BUNDLE / SSL_CERT_FILE: extra trust anchor for a
    # self-signed MCP endpoint (e.g. a self-hosted hybrid VM's Caddy `tls internal`
    # cert). `create_mcp_http_client` below builds a plain httpx.AsyncClient with
    # no `verify=` kwarg, so httpx falls back to its OWN default (certifi's bundle)
    # for `verify=True` and â€” unlike stdlib ssl or `requests` â€” does NOT read
    # SSL_CERT_FILE/REQUESTS_CA_BUNDLE from the environment on its own. A fresh
    # agent with no manually-trusted CA therefore fails TLS verification before any
    # request is even sent (confirmed live elsewhere: a plain httpx.AsyncClient()
    # against a self-signed endpoint raises SSLCertVerificationError: unable to get
    # local issuer certificate â€” the generic "unhandled errors in a TaskGroup"
    # callers see is just that SSL error re-wrapped by anyio's TaskGroup, with no
    # request ever reaching the server). Passing verify=<path> explicitly is the
    # only way to make httpx honor this trust anchor; unset (the default â€” MCP_SERVER_URL
    # here is the real prod SaaS endpoint with a publicly-trusted cert), this is a
    # no-op and behavior is identical to before.
    ca_bundle = (
        os.environ.get("MCP_CA_BUNDLE")
        or os.environ.get("REQUESTS_CA_BUNDLE")
        or os.environ.get("SSL_CERT_FILE")
        or ""
    ).strip()
    # Opt-in only (unlike some other scripts that default this to insecure) â€”
    # this file is also used for real customer runs against the actual SaaS
    # endpoint, so TLS verification must stay on unless explicitly disabled
    # for a known, private endpoint (e.g. a self-signed demo VM).
    insecure = (os.environ.get("MCP_TLS_INSECURE_SKIP_VERIFY", "") or "").strip().lower() in ("1", "true", "yes")

    if httpx_client_factory is not None:
        client = httpx_client_factory(headers=headers, timeout=httpx_timeout, auth=auth)
    elif ca_bundle:
        client = httpx.AsyncClient(
            follow_redirects=True, headers=headers, timeout=httpx_timeout, auth=auth, verify=ca_bundle,
        )
    elif insecure:
        logger.warning(
            "MCP_TLS_INSECURE_SKIP_VERIFY is set â€” TLS certificate and hostname "
            "verification are DISABLED for the MCP connection. Do not use this "
            "against anything but a known, private endpoint."
        )
        client = httpx.AsyncClient(
            follow_redirects=True, headers=headers, timeout=httpx_timeout, auth=auth, verify=False,
        )
    else:
        client = create_mcp_http_client(headers=headers, timeout=httpx_timeout, auth=auth)

    async with client:
        async with streamable_http_client(
            url, http_client=client, terminate_on_close=terminate_on_close,
        ) as streams:
            if len(streams) == 2:
                streams = (*streams, None)
            yield streams


def _upload_to_s3(presigned_url: str, archive_path: str) -> None:
    size = os.path.getsize(archive_path)
    logger.debug("Uploading %d KB to S3 ...", size // 1024)
    with open(archive_path, "rb") as f:
        req = urllib.request.Request(
            presigned_url, data=f.read(), method="PUT",
            headers={"Content-Type": "application/zip"},
        )
        with urllib.request.urlopen(req) as resp:
            if resp.status not in (200, 204):
                raise RuntimeError(f"S3 upload failed: HTTP {resp.status}")
    logger.debug("S3 upload complete")


def _is_payload_too_large(exc: BaseException) -> bool:
    """True when exc is (or wraps) an HTTP 413 from the MCP endpoint.

    The ``mcp`` SDK's StreamableHTTPSessionManager hard-caps request bodies at
    4 MiB and 413s anything bigger before it ever reaches application code â€”
    archive_content_base64 (used in hybrid mode, where source never touches
    S3) routes the whole archive through that same request body, so a batch
    whose files happen to be large enough hits this cap. Retrying the
    identical payload would just 413 again; the caller should split the batch
    in half instead.
    """
    cause = exc
    while hasattr(cause, "exceptions") and cause.exceptions:
        cause = cause.exceptions[0]
    try:
        import httpx
        if isinstance(cause, httpx.HTTPStatusError) and cause.response is not None:
            return cause.response.status_code == 413
    except Exception:
        pass
    text = str(cause)
    return "413" in text and ("Request Entity Too Large" in text or "Request body too large" in text)


def _loads_scan_payload(raw: str) -> dict:
    """Parse plain JSON or report-first MCP scan text (stdlib only).
    """
    text = (raw or "").strip()
    if not text:
        return {"raw": "empty response"}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    marker = "<!-- LINEAJE_MCP_JSON -->"
    if marker in text:
        json_part = text.split(marker, 1)[1].strip()
        try:
            parsed = json.loads(json_part)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    brace = text.rfind("\n{")
    if brace >= 0:
        try:
            parsed = json.loads(text[brace + 1:])
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return {"raw": text, "report": text}


def _parse_tool_result(result: Any) -> dict:
    for attr in ("structuredContent", "structured_content"):
        structured = getattr(result, attr, None)
        if isinstance(structured, dict) and (
            "report" in structured or "status" in structured or "error" in structured
        ):
            return structured
    if hasattr(result, "content") and result.content:
        raw = result.content[0].text if hasattr(result.content[0], "text") else str(result.content[0])
        return _loads_scan_payload(raw)
    return {"raw": "empty response"}


def _describe_exception(exc: BaseException) -> str:
    """Unwrap ExceptionGroup/BaseExceptionGroup (e.g. from asyncio.TaskGroup, which
    anyio's streamablehttp_client uses internally) down to the real underlying
    message(s). Without this, a caller only ever sees the generic
    "unhandled errors in a TaskGroup (N sub-exception)" wrapper text â€” which
    hides exactly the RuntimeError this script raises on purpose (auth
    failures, get_upload_url errors, etc.) or the SSL error a bad/missing
    MCP_CA_BUNDLE produces. BaseExceptionGroup is a builtin since Python 3.11,
    the same version that introduced asyncio.TaskGroup, so no import/version
    guard is needed here."""
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(_describe_exception(e) for e in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"


# Guardrail stub insertion
_GR_CHECK_MARKERS: Dict[str, str] = {
    ".py": "def gr_check(",
    ".js": "function gr_check(",
    ".jsx": "function gr_check(",
    ".ts": "function gr_check(",
    ".tsx": "function gr_check(",
    ".go": "func grCheck(",
    ".java": "class GrClient {",
}

_GUARDRAIL_MANIFEST_REL = ".env.example"


def _module_prefix_insert_index(lines: List[str]) -> int:
    """0-indexed position to insert a new top-level block at â€” after any
    shebang/encoding comment AND after a leading module docstring, if
    present. Only the first statement in a Python file is its module
    docstring; inserting above it silently demotes it to a dead string-
    literal expression. Falls back to shebang/encoding-only detection for
    non-Python sources â€” never raises."""
    idx = 0
    for i, ln in enumerate(lines[:5]):
        if ln.startswith("#!") or ln.strip().startswith("# -*-"):
            idx = i + 1
        if ln.startswith("package "):
            idx = max(idx, i + 1)
    try:
        tree = ast.parse("".join(lines))
        first = tree.body[0] if tree.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(getattr(first, "value", None), ast.Constant)
            and isinstance(first.value.value, str)
        ):
            idx = max(idx, first.end_lineno)
    except SyntaxError:
        pass
    return idx


def safe_prefix_insert_index(lines: List[str]) -> int:
    """Like _module_prefix_insert_index(), but also walks past any leading
    `from __future__ import` line(s) â€” those must be the first statement(s)
    in a Python file; inserting anything above one is a real SyntaxError."""
    insert_at = _module_prefix_insert_index(lines)
    while insert_at < len(lines) and lines[insert_at].lstrip().startswith("from __future__ import"):
        insert_at += 1
    return insert_at


def validate_python_source(new_content: str, abs_path: str) -> Optional[str]:
    """Whole-file compile() check after a stub insertion. Returns None if
    still valid, else the SyntaxError message. compile(), not ast.parse() â€”
    ast.parse() does not enforce future-import placement."""
    try:
        compile(new_content, abs_path, "exec")
        return None
    except SyntaxError as exc:
        return str(exc)


def _norm_stub_relpath(path: str) -> str:
    """Repo-relative path used only for stub-identity comparison."""
    p = (path or "").replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    p = p.lstrip("/")
    if os.path.basename(p) == "gr_stub_client.py":
        return "gr_stub_client.py"
    return p


def _stub_identity_keys(row: Dict[str, Any]) -> list:
    rel = _norm_stub_relpath(str(row.get("file") or ""))
    try:
        line = int(row.get("line") or 0)
    except (TypeError, ValueError):
        line = 0
    ip = str(row.get("insertion_point") or "")
    site = str(row.get("site_id") or "")
    keys: list = []
    if site:
        keys.append(("site", site))
    if rel:
        keys.append(("loc", rel, line, ip))
        if line == 0 and not ip:
            keys.append(("file", rel))
    return keys


def _dedupe_stub_insertions(stubs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse overlapping batches / path aliases into one row per site."""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for s in stubs:
        if not isinstance(s, dict):
            out.append(s)
            continue
        entry = dict(s)
        rel = _norm_stub_relpath(str(s.get("file") or ""))
        if rel:
            entry["file"] = rel
        keys = _stub_identity_keys(entry)
        if keys and any(k in seen for k in keys):
            if entry.get("new_content"):
                for i, kept in enumerate(out):
                    if not isinstance(kept, dict):
                        continue
                    kept_keys = _stub_identity_keys(kept)
                    if any(k in kept_keys for k in keys):
                        merged = dict(kept)
                        merged["new_content"] = entry["new_content"]
                        merged["status"] = "detected"
                        merged["safe_to_insert"] = True
                        merged["file"] = rel or merged.get("file") or ""
                        out[i] = merged
                        break
            continue
        for k in keys:
            seen.add(k)
        out.append(entry)
    return out


def _collapse_stub_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_sites: set = set()
    seen_loc: set = set()
    out: List[Dict[str, Any]] = []
    for hit in sorted(hits, key=lambda h: int(h.get("line") or 0)):
        site = str(hit.get("site_id") or "")
        try:
            line = int(hit.get("line") or 0)
        except (TypeError, ValueError):
            line = 0
        loc = (line, str(hit.get("insertion_point") or ""))
        if site and site in seen_sites:
            continue
        if loc in seen_loc:
            continue
        if site:
            seen_sites.add(site)
        seen_loc.add(loc)
        out.append(hit)
    return out


def _stub_insertions_from_mcp_result(mcp_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Merge stub_insertions with patched_files / companion_files from the MCP response."""
    stubs = [dict(s) for s in (mcp_result.get("stub_insertions") or [])]
    by_file: Dict[str, str] = {}
    for extra in list(mcp_result.get("patched_files") or []) + list(mcp_result.get("companion_files") or []):
        if not isinstance(extra, dict):
            continue
        rel = _norm_stub_relpath(extra.get("file") or "")
        if rel and extra.get("content") is not None:
            by_file[rel] = extra["content"]
    if not by_file:
        return _dedupe_stub_insertions(stubs)
    applied: set = set()
    for s in stubs:
        rel = _norm_stub_relpath(s.get("file") or "")
        if rel:
            s["file"] = rel
        if rel in by_file:
            s["new_content"] = by_file[rel]
            s["status"] = "detected"
            s["safe_to_insert"] = True
            applied.add(rel)
    for rel, content in by_file.items():
        if rel not in applied:
            stubs.append({
                "file": rel,
                "status": "detected",
                "safe_to_insert": True,
                "new_content": content,
                "line": 0,
            })
    return _dedupe_stub_insertions(stubs)


def apply_stub_insertions_to_clone(
    stub_insertions: List[Dict[str, Any]],
    source_dir: str,
) -> Dict[str, str]:
    """Apply server-computed stub_insertions. Returns repo-relative path â†’ content.

    Unsafe / missing / invalid hits are dropped silently â€” they are not logged
    and are not returned as skipped/failed rows.
    """
    by_file: Dict[str, List[Dict[str, Any]]] = {}
    validated: Dict[str, str] = {}
    for s in _dedupe_stub_insertions(stub_insertions):
        rel = _norm_stub_relpath(s.get("file") or "")
        if not rel:
            continue
        if s.get("status") != "detected":
            continue  # "already_present" â€” nothing to do
        if s.get("new_content"):
            # Server already instrumented the extracted archive â€” write the
            # whole file rather than re-applying proposed_stub line-by-line.
            validated[rel] = s["new_content"]
            continue
        if rel in validated:
            continue
        if not s.get("safe_to_insert"):
            continue
        by_file.setdefault(rel, []).append(s)

    for rel_path, hits in by_file.items():
        if rel_path in validated:
            continue
        abs_path = os.path.join(source_dir, rel_path)
        if not os.path.isfile(abs_path):
            continue

        with open(abs_path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        ext = pathlib.Path(rel_path).suffix.lower()

        sorted_hits = sorted(_collapse_stub_hits(hits), key=lambda h: h.get("line", 0), reverse=True)
        needs_import = False
        for hit in sorted_hits:
            proposed = hit.get("proposed_stub") or ""
            if not proposed:
                continue
            site = str(hit.get("site_id") or "")
            joined = "".join(lines)
            marker = f"site_id={site!r}" if site else ""
            if (marker and marker in joined) or proposed in joined:
                continue
            line = int(hit.get("line") or 0)
            # 1-based line; insert_after=True (result/lhs patterns) must land
            # AFTER the line assigning the variable the stub references â€”
            # inserting before it is a NameError.
            idx = max(0, line if hit.get("insert_after") else line - 1)
            idx = min(idx, len(lines))
            lines.insert(idx, proposed + "\n")
            needs_import = True

        if needs_import:
            import_block = next(
                (h.get("import_needed") for h in hits if h.get("import_needed")), "",
            )
            marker = _GR_CHECK_MARKERS.get(ext, "def gr_check(")
            if import_block and marker not in "".join(lines):
                lines.insert(safe_prefix_insert_index(lines), import_block.rstrip("\n") + "\n")

        new_content = "".join(lines)
        if ext == ".py":
            syntax_err = validate_python_source(new_content, abs_path)
            if syntax_err:
                continue

        validated[rel_path] = new_content

    if validated:
        logger.info("Applied guardrail stubs in %d file(s)", len(validated))
    return validated


_HARDCODED_GR_ORIGIN = (
    lambda _p: f"{_p.scheme}://{_p.netloc}" if _p.scheme and _p.netloc else ""
)(urllib.parse.urlsplit(MCP_SERVER_URL))
_ENV_EXAMPLE_HEADER = (
    "# Lineaje UnifAI guardrail stub runtime configuration.\n"
    "# Generated by the Lineaje hybrid scan â€” read directly by gr_stub_client.py's\n"
    "# env-var overrides at POST /enforce time.\n"
    "# If the Lineaje GR/guardrail service runs on the same VM as this codebase,\n"
    "# use https://localhost/enforce instead of a public IP or hostname.\n"
    "\n"
)


def _usable_scan_refresh_token(raw: str) -> str:
    """SCIM refresh token only â€” never a JWT, URL, or identity ``lineaje_pat_``."""
    s = _normalize_token(raw)
    if not s or s.startswith("http"):
        return ""
    if s.count(".") == 2 and s.startswith("eyJ"):
        return ""
    if s.startswith("lineaje_pat"):
        return ""
    return s


def _origin_for_gr_manifest(server_url: str) -> str:
    """scheme://host[:port] this scan actually connected to, minus any path
    (e.g. the ``/mcp`` in ``--mcp-server-url``) â€” or "" if unusable (missing,
    unparseable, or loopback: a customer's own runtime guardrail stub cannot
    reach "localhost" meaning this CI runner)."""
    s = (server_url or "").strip()
    if not s:
        return ""
    try:
        parsed = urllib.parse.urlsplit(s)
    except ValueError:
        return ""
    if not parsed.scheme or not parsed.netloc:
        return ""
    host = (parsed.hostname or "").strip().lower()
    if host in ("localhost", "127.0.0.1", "::1") or host.startswith("127."):
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def _upsert_env_file_content(existing_text: str, updates: Dict[str, str], header: str) -> str:
    """Write or update ``KEY=VALUE`` entries in ``.env``-style text.

    Mirrors mcp_server.py's ``_upsert_env_file`` minus filesystem I/O â€”
    ``validated_fixes`` only ever holds content in memory here.
    """
    lines = existing_text.splitlines() if existing_text.strip() else header.splitlines()
    written_keys: set = set()
    new_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            new_lines.append(f"{key}={updates[key]}")
            written_keys.add(key)
        else:
            new_lines.append(line)
    for key, val in updates.items():
        if key not in written_keys:
            new_lines.append(f"{key}={val}")
    return "\n".join(new_lines) + "\n"


def _ensure_env_example_in_validated_fixes(
    validated_fixes: Dict[str, str],
    source_dir: str = "",
    refresh_token: str = "",
    server_url: str = "",
) -> None:
    """Write this scan's own SCIM refresh token + MCP server origin into
    customer ``.env.example`` as ``GR_SERVICE_URL`` / ``LINEAJE_REFRESH_TOKEN``
    â€” the exact env vars ``gr_stub_client.py`` reads as overrides at
    POST /enforce time. Mirrors ``gha_repo_scan.py``'s
    ``_ensure_refresh_token_in_validated_fixes`` and mcp_server.py's
    ``_write_guardrail_runtime_manifest`` so all three insertion paths ship
    the same runtime config file.
    """
    rel = _GUARDRAIL_MANIFEST_REL
    rt = _usable_scan_refresh_token(refresh_token)
    existing_text = validated_fixes.get(rel, "")
    m = re.search(r"^LINEAJE_REFRESH_TOKEN=(.*)$", existing_text, re.MULTILINE)
    # This scan's own credential (rt) wins over whatever was already written
    # (keep) â€” a pre-existing value here is at best a separately-minted
    # credential for some other tenant, never the one that actually ran this
    # scan. Only fall back to it when this script somehow has no usable
    # token of its own.
    keep = _usable_scan_refresh_token(m.group(1).strip() if m else "")
    token = rt or keep
    if not token:
        logger.warning(
            "No SCIM refresh token for %s â€” set LINEAJE_PAT_TOKEN "
            "(do not store a JWT or lineaje_pat_ identity PAT)",
            rel,
        )
        return
    # Likewise: the MCP server URL this scan actually used (minus its /mcp
    # path) wins over the hardcoded hosted SaaS origin â€” a customer's own
    # runtime guardrail stub cannot reach "localhost" meaning this runner,
    # which is why --mcp-server-url is sometimes pointed at localhost/a
    # private IP to route around Azure's public-IP hairpin-NAT limitation.
    base = _origin_for_gr_manifest(server_url) or _HARDCODED_GR_ORIGIN
    updated = _upsert_env_file_content(
        existing_text,
        {"GR_SERVICE_URL": f"{base}/enforce", "LINEAJE_REFRESH_TOKEN": token},
        _ENV_EXAMPLE_HEADER,
    )
    validated_fixes[rel] = updated
    if source_dir:
        dest = os.path.join(source_dir, rel)
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "w", encoding="utf-8") as fh:
                fh.write(updated)
        except OSError as exc:
            logger.warning("Could not write %s: %s", dest, exc)
    logger.info("Customer %s LINEAJE_REFRESH_TOKEN=set (SCIM refresh token)", rel)


def _run_mcp_scan_via_client(
    server_url: str,
    bearer_getter: Callable[[], str],
    source_code_repo: str,
    branch: str,
    files_to_scan: List[str],
    archive_path: str,
    head_sha: str = "",
    is_last_batch: bool = True,
    sbom_id: str = "",
    run_id: str = "",
) -> Dict[str, Any]:
    from mcp import ClientSession

    async def _scan() -> Dict[str, Any]:
        base_args: Dict[str, Any] = {
            "source_code_repo": source_code_repo,
            "branch_or_tag": branch,
            "files_to_scan": files_to_scan,
        }
        upload_args: Dict[str, Any] = dict(base_args)
        resolved_sbom = (sbom_id or "").strip()
        if resolved_sbom:
            upload_args["sbom_id"] = resolved_sbom
        resolved_run_id = (run_id or "").strip()
        if resolved_run_id:
            upload_args["run_id"] = resolved_run_id
        with open(archive_path, "rb") as _fh:
            upload_args["archive_content_base64"] = base64.b64encode(_fh.read()).decode("ascii")
        scm_headers: Dict[str, str] = {"X-Unifai-Commit-Sha": head_sha} if head_sha else {}

        tok1 = bearer_getter()
        async with streamablehttp_client(
            server_url, headers={"Authorization": f"Bearer {tok1}", **scm_headers},
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                # logger.info("MCP step 1/3: get_upload_url")
                upload_result = _parse_tool_result(
                    await session.call_tool("get_upload_url", arguments=upload_args)
                )
                if not upload_result.get("success"):
                    raise RuntimeError(f"get_upload_url failed: {upload_result.get('error', upload_result)}")
                archive_id = upload_result["archive_id"]
                presigned_url = upload_result["presigned_url"]
                resolved_sbom = (upload_result.get("sbom_id") or resolved_sbom or "").strip()
                resolved_run_id = (upload_result.get("run_id") or resolved_run_id or "").strip()
                # False when a hybrid server kept the source on its own host; missing
                # (None) on servers older than that flag.
                source_s3_uploaded = upload_result.get("source_archive_s3_uploaded")


        if presigned_url:
            # logger.info("MCP step 2/3: upload to S3")
            _upload_to_s3(presigned_url, archive_path)

        tok2 = bearer_getter()
        sse_timeout = int(os.environ.get("UNIFAI_MCP_SSE_READ_TIMEOUT", "1800"))
        async with streamablehttp_client(
            server_url,
            headers={"Authorization": f"Bearer {tok2}", **scm_headers},
            sse_read_timeout=sse_timeout,
        ) as (read2, write2, _):
            async with ClientSession(read2, write2) as session2:
                await session2.initialize()
                # logger.info("MCP step 3/3: analyze_uploaded_archive (timeout=%ds)", sse_timeout)
                analyze_args = dict(base_args)
                analyze_args["archive_id"] = archive_id
                # This script's own URL to reach the MCP server â€” more reliable
                # than any guess the server could make about its own address.
                analyze_args["mcp_server_location"] = server_url
                # Required so evidence.source_metadata.evidence_type is scm_scan
                # (the server docstring: "Scripts and GitHub Actions MUST pass
                # scm_scan") instead of defaulting to ide_scan.
                analyze_args["scan_type"] = "scm_scan"
                analyze_args["is_last_batch"] = is_last_batch
                if resolved_sbom:
                    analyze_args["sbom_id"] = resolved_sbom
                if resolved_run_id:
                    analyze_args["run_id"] = resolved_run_id
                result = _parse_tool_result(
                    await session2.call_tool("analyze_uploaded_archive", arguments=analyze_args)
                )
                if resolved_sbom and not (result.get("sbom_id") or "").strip():
                    result["sbom_id"] = resolved_sbom
                if resolved_run_id and not (result.get("run_id") or "").strip():
                    result["run_id"] = resolved_run_id
                result["source_archive_s3_uploaded"] = (
                    True if presigned_url else source_s3_uploaded
                )
                return result

    return asyncio.run(_scan())


def run_mcp_scan(
    server_url: str,
    bearer_getter: Callable[[], str],
    source_code_repo: str,
    branch: str,
    files_to_scan: List[str],
    archive_path: str,
    head_sha: str = "",
    is_last_batch: bool = True,
    sbom_id: str = "",
    run_id: str = "",
) -> Dict[str, Any]:
    logger.info("MCP scan: %d files, repo=%s, branch=%s", len(files_to_scan), source_code_repo, branch)
    return _run_mcp_scan_via_client(
        server_url, bearer_getter, source_code_repo, branch, files_to_scan, archive_path,
        head_sha=head_sha, is_last_batch=is_last_batch, sbom_id=sbom_id, run_id=run_id,
    )

# ===========================================================================
# Parallel batch scan
# ===========================================================================

def parallel_batch_scan(
    batches: List[List[str]],
    source_dir: str,
    temp_dir: str,
    source_code_repo: str,
    branch: str,
    head_sha: str,
    run_id: str,
    server_url: str,
    bearer_getter: Callable[[], str],
    max_workers: int = MAX_SCAN_WORKERS,
    manifest_files: Optional[List[str]] = None,
) -> Tuple[
    List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[Dict[str, str]],
    int, List[str], List[Dict[str, Any]], Dict[str, Any],
]:
    all_violations: List[Dict[str, Any]] = []
    all_remediation_actions: List[Dict[str, Any]] = []
    all_reports: List[str] = []
    all_aibom: List[Dict[str, str]] = []
    all_stub_insertions: List[Dict[str, Any]] = []
    # What the server PUT to Lineaje S3, plus whether any batch's source archive went too.
    uploaded: Dict[str, Any] = {"entities": [], "findings": [], "source_archive_s3_uploaded": None}
    aibom_seen: set = set()
    failed_batch_count = 0
    failure_details: List[str] = []
    lock = threading.Lock()
    scan_sbom_id = ""

    def _scan_leaf(label: str, batch_files: List[str]) -> Dict[str, Any]:
        nonlocal scan_sbom_id
        logger.info("Batch %s/%d: %d files", label, len(batches), len(batch_files))
        archive_path = create_batch_archive(
            source_dir, temp_dir, batch_files,
            source_code_repo, branch, head_sha, label, run_id=run_id,
            manifest_files=manifest_files if str(label) == "1" else None,
        )
        result = run_mcp_scan(
            server_url, bearer_getter, source_code_repo, branch, batch_files, archive_path,
            head_sha=head_sha, is_last_batch=True, sbom_id=scan_sbom_id, run_id=run_id,
        )
        with lock:
            resolved_sbom = (result.get("sbom_id") or "").strip()
            if resolved_sbom and not scan_sbom_id:
                scan_sbom_id = resolved_sbom
        return result

    def _scan_one(batch_idx: int, batch_files: List[str]) -> Tuple[int, List[Tuple[str, Dict[str, Any]]]]:
        """Scan a top-level batch, splitting in half and retrying on 413
        (request too large for the mcp SDK's 4 MiB body cap) until every leaf
        either succeeds or can't be split further. Returns one (label,
        result) pair per leaf batch that was actually sent."""

        def _run(label: str, files: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
            try:
                return [(label, _scan_leaf(label, files))]
            except BaseException as exc:
                split = split_batch_keeping_manifests_whole(files) if _is_payload_too_large(exc) else None
                if not split:
                    raise
                first_half, second_half = split
                logger.warning(
                    "Batch %s: 413 Request Entity Too Large (%d files) â€” "
                    "splitting into %d + %d files and retrying each half",
                    label, len(files), len(first_half), len(second_half),
                )
                return _run(f"{label}a", first_half) + _run(f"{label}b", second_half)

        return batch_idx, _run(str(batch_idx), batch_files)

    def _collect(label: str, mcp_result: Dict[str, Any]) -> None:
        batch_actions = mcp_result.get("remediation_actions", [])
        batch_violations = list(mcp_result.get("violations") or [])
        if not batch_violations:
            batch_violations = [
                {k: v for k, v in a.items() if k != "fix_code"}
                for a in batch_actions
                if a.get("file")
            ]
        batch_report = mcp_result.get("report", "")
        batch_aibom = mcp_result.get("aibom", [])
        batch_stub_insertions = _stub_insertions_from_mcp_result(mcp_result)
        logger.info(
            "Batch %s/%d done: status=%s violations=%d aibom=%d stub_insertions=%d",
            label, len(batches), mcp_result.get("status", "unknown"),
            len(batch_violations), len(batch_aibom), len(batch_stub_insertions),
        )
        with lock:
            all_violations.extend(batch_violations)
            all_remediation_actions.extend(batch_actions)
            all_stub_insertions.extend(batch_stub_insertions)
            if batch_report:
                all_reports.append(batch_report)
            uploaded["entities"].extend(mcp_result.get("uploaded_entities_json") or [])
            uploaded["findings"].extend(mcp_result.get("uploaded_findings_json") or [])
            src_flag = mcp_result.get("source_archive_s3_uploaded")
            if src_flag is not None:
                uploaded["source_archive_s3_uploaded"] = bool(
                    uploaded["source_archive_s3_uploaded"]) or bool(src_flag)
            for entry in batch_aibom:
                key = (entry.get("name", ""), entry.get("source_file", ""))
                if key not in aibom_seen:
                    aibom_seen.add(key)
                    all_aibom.append(entry)

    workers = min(len(batches), max_workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_scan_one, idx, files): idx for idx, files in enumerate(batches, 1)}
        for future in as_completed(future_map):
            batch_idx = future_map[future]
            try:
                _, leaf_results = future.result()
                for label, mcp_result in leaf_results:
                    _collect(label, mcp_result)
            except BaseException as exc:
                failed_batch_count += 1
                detail = f"Batch {batch_idx}/{len(batches)} failed: {_describe_exception(exc)}"
                logger.error("%s", detail)
                failure_details.append(detail)

    return (
        all_violations, all_remediation_actions, all_reports, all_aibom,
        failed_batch_count, failure_details, all_stub_insertions, uploaded,
    )

# ===========================================================================
# Report merging â€” combine each batch's own markdown report (server-rendered,
# one full "# LINEAJE AI POLICY REPORT" per batch) into a single report with
# one Section 1 / 2 / 3 and one Enforcement Summary, instead of N reports
# concatenated back-to-back (one per batch).
# ===========================================================================

_REPORT_SECTION_HEADERS = [
    "## Enforcement Summary",
    "### SECTION 1: AIBOM Discovery",
    "### SECTION 2: Policy Violations",
    "### SECTION 3: Controls Enforced",
    "### AI Component Relationship Graph",
    "### Phase Timing",
]

_ENFORCEMENT_PREVIEW_CAP = 12


def _split_report_sections(report: str) -> Dict[str, str]:
    """*report* â†’ {header: content up to the next known header}, plus the
    text before the first header under ``__preamble__``."""
    positions = sorted(
        (idx, h) for h in _REPORT_SECTION_HEADERS for idx in [report.find(h)] if idx != -1
    )
    sections: Dict[str, str] = {"__preamble__": report[: positions[0][0]] if positions else report}
    for i, (idx, h) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(report)
        sections[h] = report[idx + len(h): end]
    return sections


def _table_rows(block: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    """First markdown table in *block* â†’ (header row, separator row, data rows)."""
    header = sep = None
    data: List[str] = []
    for line in block.splitlines():
        s = line.rstrip()
        if not s.lstrip().startswith("|"):
            continue
        if header is None:
            header = s
        elif sep is None:
            sep = s
        else:
            data.append(s)
    return header, sep, data


def _enforcement_bullet_lines(block: str) -> List[str]:
    """Bullet lines in an Enforcement Summary block â€” anything that isn't the
    italic "â€¦and N more" line, the "**Summary:**" line, or a table row."""
    return [
        line.strip() for line in block.splitlines()
        if line.strip() and not line.strip().startswith(("*", "#", "|"))
    ]


def _mermaid_body(block: str) -> str:
    m = re.search(r"```mermaid\n(.*?)```", block, re.DOTALL)
    return m.group(1) if m else ""


def _merge_mermaid_bodies(bodies: List[str]) -> str:
    """Concatenate per-batch mermaid graphs into one, renaming node ids per
    batch (``model_1`` â†’ ``b0_model_1``) so batches never collide, and
    deduping the (static, identical every time) ``classDef`` lines.

    Node ids can appear more than once per line â€” as both endpoints of an
    edge (``agent_4 -->|uses| model_7``), or as the subject of a ``class``
    assignment â€” so renaming has to replace every whole-word occurrence of
    a batch's declared ids in its own lines, not just a line's leading token.
    """
    node_def_lines: List[str] = []
    other_lines: List[str] = []
    classdef_lines: List[str] = []
    seen_classdef: set = set()
    for batch_idx, body in enumerate(bodies):
        declared_ids = re.findall(r"^\s*(\w+)\s*[\[\(\{]", body, re.MULTILINE)
        rename = {nid: f"b{batch_idx}_{nid}" for nid in declared_ids}
        id_pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in rename) + r")\b") if rename else None

        for raw_line in body.splitlines():
            s = raw_line.strip()
            if not s or s == "graph TD":
                continue
            if s.startswith("classDef"):
                if s not in seen_classdef:
                    seen_classdef.add(s)
                    classdef_lines.append(s)
                continue
            renamed = id_pattern.sub(lambda m: rename[m.group(1)], s) if id_pattern else s
            if re.match(r"^\w+\s*[\[\(\{]", s):
                node_def_lines.append(renamed)
            else:
                other_lines.append(renamed)
    lines = ["graph TD"] + [f"    {l}" for l in node_def_lines] + [f"    {l}" for l in other_lines]
    if classdef_lines:
        lines.append("")
        lines.extend(f"    {l}" for l in classdef_lines)
    return "\n".join(lines)


def _dedupe_preserve_order(rows: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for r in rows:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _merge_batch_reports(
    reports: List[str],
    *,
    total_violations: int,
    total_remediation_actions: int,
    total_elapsed: float,
) -> str:
    """Combine each batch's own full markdown report into one report with a
    single Enforcement Summary, Section 1 (AIBOM Discovery), Section 2
    (Policy Violations), Section 3 (Controls Enforced), relationship graph,
    and phase timing â€” rather than showing one full report per batch.
    """
    reports = [r for r in reports if r and r.strip()]
    if not reports:
        return ""
    if len(reports) == 1:
        return reports[0]

    project_scanned = ""
    enforcement_bullets: List[str] = []
    s1_header = s1_sep = None
    s1_rows: List[str] = []
    s2_header = s2_sep = None
    s2_rows: List[str] = []
    s3_header = s3_sep = None
    s3_rows: List[str] = []
    mermaid_bodies: List[str] = []
    phase_totals: Dict[str, float] = {}
    phase_order: List[str] = []

    for report in reports:
        sections = _split_report_sections(report)
        if not project_scanned:
            m = re.search(r"\*\*Project Scanned:\*\*\s*`([^`]*)`", sections.get("__preamble__", ""))
            if m:
                project_scanned = m.group(1)

        enforcement_bullets.extend(_enforcement_bullet_lines(sections.get("## Enforcement Summary", "")))

        h, sep, rows = _table_rows(sections.get("### SECTION 1: AIBOM Discovery", ""))
        s1_header, s1_sep = s1_header or h, s1_sep or sep
        s1_rows.extend(rows)

        h, sep, rows = _table_rows(sections.get("### SECTION 2: Policy Violations", ""))
        s2_header, s2_sep = s2_header or h, s2_sep or sep
        s2_rows.extend(rows)

        h, sep, rows = _table_rows(sections.get("### SECTION 3: Controls Enforced", ""))
        s3_header, s3_sep = s3_header or h, s3_sep or sep
        s3_rows.extend(rows)

        mermaid_bodies.append(_mermaid_body(sections.get("### AI Component Relationship Graph", "")))

        _, _, phase_rows = _table_rows(sections.get("### Phase Timing", ""))
        for row in phase_rows:
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            if len(cells) != 2:
                continue
            phase = cells[0].strip("*").strip()
            m = re.match(r"([\d.]+)", cells[1].strip("*").strip())
            if not m or phase.lower() == "total":
                continue
            if phase not in phase_totals:
                phase_order.append(phase)
            phase_totals[phase] = phase_totals.get(phase, 0.0) + float(m.group(1))

    s1_rows = _dedupe_preserve_order(s1_rows)
    s2_rows = _dedupe_preserve_order(s2_rows)
    s3_rows = _dedupe_preserve_order(s3_rows)

    lines: List[str] = ["# LINEAJE AI POLICY REPORT", ""]
    status = "violations_found" if total_violations else "compliant"
    lines.append(
        f"**Run summary:** `{status}` Â· violations={total_violations} Â· "
        f"remediation_actions={total_remediation_actions} Â· elapsed={total_elapsed:.1f}s"
    )
    lines.append("")
    if project_scanned:
        lines.append(f"**Project Scanned:** `{project_scanned}`")
        lines.append("")
    lines.append(f"**Total Time:** {total_elapsed:.1f}s")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Enforcement Summary")
    lines.append("")
    preview = enforcement_bullets[:_ENFORCEMENT_PREVIEW_CAP]
    for bullet in preview:
        lines.append(bullet)
        lines.append("")
    remaining = total_violations - len(preview)
    if remaining > 0:
        lines.append(f"*â€¦ and {remaining} more violation(s) â€” see **SECTION 3: Controls Enforced** below.*")
        lines.append("")
    lines.append(f"**Summary:** {len(preview)} notified")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### SECTION 1: AIBOM Discovery")
    lines.append("")
    lines.append("#### AI Components")
    lines.append("")
    if s1_header:
        lines.extend([s1_header, s1_sep, *s1_rows])
    lines.append("")
    lines.append("### SECTION 2: Policy Violations")
    lines.append("")
    lines.append("*Each row is one **`policy_violation=true`** finding (file Ã— policy).*")
    lines.append("")
    if s2_header:
        lines.extend([s2_header, s2_sep, *s2_rows])
    lines.append("")
    lines.append("### SECTION 3: Controls Enforced")
    lines.append("")
    lines.append("#### Controls enforced")
    lines.append("")
    if s3_header:
        lines.extend([s3_header, s3_sep, *s3_rows])
    lines.append("")
    lines.append("### AI Component Relationship Graph")
    lines.append("")
    lines.append("```mermaid")
    lines.append(_merge_mermaid_bodies(mermaid_bodies))
    lines.append("```")
    lines.append("")
    lines.append("### Phase Timing")
    lines.append("")
    lines.append("| Phase | Time |")
    lines.append("|-------|------|")
    for phase in phase_order:
        lines.append(f"| {phase} | {phase_totals[phase]:.1f}s |")
    lines.append(f"| **Total** | **{sum(phase_totals.values()):.1f}s** |")
    lines.append("")

    return "\n".join(lines)

# ===========================================================================
# JSON output
# ===========================================================================

def build_json_output(
    *,
    status: str,
    repo: str,
    branch: str,
    head_sha: str,
    source_code_repo: str,
    files_scanned: int,
    batches: int,
    failed_batches: int,
    violations: List[Dict[str, Any]],
    aibom: Optional[List[Dict[str, str]]] = None,
    report: str = "",
    remediation_pr: Optional[int] = None,
    remediation_branch: str = "",
    remediation_pr_url: str = "",
    failed_remediation_files: Optional[List[str]] = None,
    scan_errors: Optional[List[str]] = None,
    remediation_actions: Optional[List[Dict[str, Any]]] = None,
    uploaded_entities: Optional[List[Dict[str, Any]]] = None,
    uploaded_findings: Optional[List[Dict[str, Any]]] = None,
    source_archive_s3_uploaded: Optional[bool] = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "scan_metadata": {
            "repo": repo,
            "branch": branch,
            "head_sha": head_sha,
            "source_code_repo": source_code_repo,
            "scanned_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "files_scanned": files_scanned,
            "batches": batches,
            "failed_batches": failed_batches,
        },
        "report": report,
        "violations": violations,
        "aibom": aibom or [],
        "remediation_pr": remediation_pr,
        "remediation_branch": remediation_branch,
        "remediation_pr_url": remediation_pr_url,
        "failed_remediation_files": failed_remediation_files or [],
        "remediation_actions": remediation_actions or [],
        "uploaded_entities": uploaded_entities or [],
        "uploaded_findings": uploaded_findings or [],
        "source_archive_s3_uploaded": source_archive_s3_uploaded,
        "scan_errors": scan_errors or [],
    }


def print_human_output(output: Dict[str, Any]) -> None:
    status = output.get("status", "unknown")
    violations = output.get("violations", [])
    scan_errors = output.get("scan_errors", [])
    metadata = output.get("scan_metadata", {})
    scanned_at = metadata.get("scanned_at", "")
    branch = metadata.get("branch", "")

    if status == "compliant":
        status_label = "âœ… Compliant"
    elif status == "violations_found":
        status_label = "âŒ Not Compliant"
    else:
        status_label = status

    logger.info("Result: %s â€” %d violation(s)", status_label, len(violations))

    print("# UnifAI Security Report")
    print()
    print(f"**Status:** {status_label}")
    if branch:
        print(f"**Branch:** `{branch}`")
    if scanned_at:
        print(f"**Scanned at:** {scanned_at}")
    rem_pr_url = output.get("remediation_pr_url", "")
    if rem_pr_url:
        print(f"**Remediation PR:** {rem_pr_url}")

    if scan_errors:
        print("\n**Errors:**")
        for err in scan_errors:
            print(f"- {err}")
        print()

    if not violations:
        if status == "compliant":
            print("\nNo violations found.")
        _print_uploaded_artifacts(output)
        return

    from collections import defaultdict
    by_file: Dict[str, List[str]] = defaultdict(list)
    for v in violations:
        file_ = v.get("file") or v.get("file_path") or "(unknown)"
        control = v.get("policy_name") or v.get("control") or v.get("policy_id") or "(unknown)"
        by_file[file_].append(control)

    num_files = len(by_file)
    print(f"\n**{len(violations)} violation(s) across {num_files} file(s)**\n")

    print("| File | Policy Violations |")
    print("|------|-------------------|")

    for file_, controls in sorted(by_file.items()):
        numbered = "".join(f"{i}. {c}<br>" for i, c in enumerate(controls, 1))
        print(f"| `{file_}` | {numbered} |")

    _print_uploaded_artifacts(output)


# GitHub caps a step summary at 1 MiB; keep each JSON well under half of that.
UPLOADED_JSON_SUMMARY_LIMIT = int(os.environ.get("UNIFAI_UPLOADED_JSON_SUMMARY_LIMIT", "400000"))


def _print_uploaded_artifacts(output: Dict[str, Any]) -> None:
    """Print the entities/findings JSON the server uploaded to Lineaje, verbatim."""
    artifacts = [
        ("entities.aientity.json", output.get("uploaded_entities") or []),
        ("findings.aifinding.json", output.get("uploaded_findings") or []),
    ]
    if not any(items for _, items in artifacts):
        return
    print("\n### Uploaded to Lineaje\n")
    src = output.get("source_archive_s3_uploaded")
    if src is False:
        print(
            "*Only these scan results were uploaded. The source code stayed on the "
            "MCP host (hybrid mode) and was not uploaded.*\n"
        )
    elif src:
        print("*The source archive was also uploaded to Lineaje (SaaS mode, or "
              "UNIFAI_FORCE_S3_ARCHIVE_UPLOAD=true).*\n")
    for name, items in artifacts:
        text = json.dumps(items, indent=2, default=str)
        truncated = len(text) > UPLOADED_JSON_SUMMARY_LIMIT
        if truncated:
            text = text[:UPLOADED_JSON_SUMMARY_LIMIT].rstrip() + "\nâ€¦ (truncated)"
        # Full copy always goes to the job log (stderr), even when the summary truncates.
        logger.info("Uploaded %s (%d item(s)):\n%s", name, len(items), json.dumps(items, indent=2, default=str))
        print(f"<details><summary><code>{name}</code> â€” {len(items)} item(s)"
              f"{' (truncated; full JSON in the job log)' if truncated else ''}</summary>\n")
        print("```json")
        print(text)
        print("```\n</details>\n")


# ===========================================================================
# Patch application (ported from veracode_repo_scan.py, no external deps)
# ===========================================================================

def _normalize_for_patch_match(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s)


def _apply_fix_entry(content: str, original: str, replacement: str) -> Tuple[str, bool]:
    if not original:
        return content, False

    if original in content:
        return content.replace(original, replacement, 1), True

    orig_stripped = original.strip()
    if orig_stripped and orig_stripped in content:
        return content.replace(orig_stripped, replacement, 1), True

    norm_orig = _normalize_for_patch_match(orig_stripped)
    norm_content = _normalize_for_patch_match(content)
    idx = norm_content.find(norm_orig)
    if idx != -1:
        orig_len = len(orig_stripped)
        real_idx = 0
        norm_walked = 0
        for ci, ch in enumerate(content):
            if norm_walked >= idx:
                real_idx = ci
                break
            norm_walked += len(_normalize_for_patch_match(ch))
        else:
            real_idx = len(content)
        sub = content[real_idx : real_idx + orig_len + 50]
        if orig_stripped in sub:
            actual_idx = content.find(orig_stripped, real_idx)
            if actual_idx != -1:
                return content[:actual_idx] + replacement + content[actual_idx + len(orig_stripped):], True

    orig_lines = [l for l in orig_stripped.splitlines() if l.strip()]
    if orig_lines:
        anchor = orig_lines[0].strip()
        if len(anchor) > 15:
            anchor_idx = content.find(anchor)
            if anchor_idx != -1:
                end_search = content.find(orig_lines[-1].strip(), anchor_idx) if len(orig_lines) > 1 else anchor_idx
                if end_search != -1:
                    end_idx = end_search + len(orig_lines[-1].strip())
                    found_block = content[anchor_idx:end_idx]
                    if len(found_block) < len(orig_stripped) * 2:
                        return content[:anchor_idx] + replacement + content[end_idx:], True

    return content, False


def _norm_rel_path(p: str) -> str:
    s = p.strip().replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    return s


def _resolve_source_file(source_dir: str, filepath: str, file_list: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a violation filepath to (rel_path, content) from the live checkout."""
    raw = filepath.strip()
    if not raw:
        return None, None
    norm_fp = _norm_rel_path(raw)
    root = pathlib.Path(source_dir)

    candidate = root / raw
    if candidate.is_file():
        return norm_fp, candidate.read_text(errors="replace")

    # Try normalised path
    candidate2 = root / norm_fp
    if candidate2.is_file():
        return norm_fp, candidate2.read_text(errors="replace")

    # Basename fallback
    base = pathlib.Path(norm_fp).name
    matches = [f for f in file_list if pathlib.Path(f).name == base]
    if len(matches) == 1:
        full = root / matches[0]
        if full.is_file():
            return _norm_rel_path(matches[0]), full.read_text(errors="replace")

    logger.warning("Cannot resolve remediation file %r in source dir", raw)
    return None, None


def apply_pipeline_fix_code_to_clone(
    remediation_actions: List[Dict[str, Any]],
    source_dir: str,
    file_list: List[str],
) -> Tuple[Dict[str, str], List[str], List[Dict[str, str]]]:
    """Apply fix_code patches from MCP remediation_actions to checked-out files.

    Returns (validated_fixes, failed_files, fix_table_rows).
    """
    validated_fixes: Dict[str, str] = {}
    failed_files: List[str] = []
    fix_table_rows: List[Dict[str, str]] = []

    by_file: Dict[str, List[Dict[str, Any]]] = {}
    for action in remediation_actions:
        fp = (action.get("file") or "").strip()
        if fp:
            by_file.setdefault(fp, []).append(action)

    for filepath, actions in by_file.items():
        has_fix_code = any(action.get("fix_code") for action in actions)
        if not has_fix_code:
            failed_files.append(filepath)
            continue

        rel_path, original_content = _resolve_source_file(source_dir, filepath, file_list)
        if rel_path is None or original_content is None:
            failed_files.append(filepath)
            continue

        content = original_content
        patch_applied = False
        for action in actions:
            for fix_entry in (action.get("fix_code") or []):
                original = fix_entry.get("original") or ""
                replacement = fix_entry.get("replacement", "")
                if not original.strip():
                    continue
                content, applied = _apply_fix_entry(content, original, replacement)
                if applied:
                    patch_applied = True
                else:
                    logger.debug(
                        "Patch not applied for %r â€” original snippet (%d chars) not found",
                        filepath, len(original),
                    )

        if patch_applied and content != original_content:
            validated_fixes[rel_path] = content
            for action in actions:
                fix_table_rows.append({
                    "policy": action.get("control", ""),
                    "description": (action.get("instruction") or "")[:200],
                    "file": filepath,
                })
        else:
            logger.warning("No patch applied for %r â€” snippets did not match file content", filepath)
            failed_files.append(filepath)

    return validated_fixes, failed_files, fix_table_rows


# ===========================================================================
# LLM remediation â€” report / PR sections
# ===========================================================================

def _md_cell(text: Any, limit: int = 300) -> str:
    t = " ".join(str(text or "").split())
    if len(t) > limit:
        t = t[: limit - 1].rstrip() + "â€¦"
    return t.replace("|", "\\|")


def _pr_remediation_details(fix_table: List[Dict[str, str]], limit_chars: int) -> str:
    """Compact per-file remediation list for the PR body (policy + what was changed)."""
    if not fix_table:
        return ""
    lines = ["### Remediation details (LLM)", "", "| File | Policy | Change |", "|------|--------|--------|"]
    for r in fix_table:
        lines.append(
            f"| `{_md_cell(r.get('file'), 100)}` | {_md_cell(r.get('policy'), 100)} | "
            f"{_md_cell(r.get('description'), 200)} |"
        )
    text = "\n".join(lines)
    if len(text) > limit_chars:
        text = text[:limit_chars].rsplit("\n", 1)[0] + "\n\n*â€¦ more rows in the scan report.*"
    return text



_PR_REPORT_SECTIONS = (
    "### SECTION 1: AIBOM Discovery",
    "### SECTION 2: Policy Violations",
    "### SECTION 3: Controls Enforced",
)


def _build_fix_pr_body(
    branch: str,
    sha_short: str,
    committed: List[str],
    failed_files: Optional[List[str]],
    fix_table: List[Dict[str, str]],
    report: str,
    limit: int,
    report_note: str = "",
) -> str:
    """Remediation PR description: SECTION 1 (AIBOM), SECTION 2 (Policy Violations),
    SECTION 3 (Controls Enforced) from the scan report, then the remediation changes.

    The remediation part is sized first so a long report is what gets truncated, never
    the list of changed files.
    """
    head = "\n".join([
        "## Lineaje AI Policy Scan",
        "",
        f"Scan of `{branch}` at `{sha_short}`. The remediation changes below are "
        "LLM-suggested fixes for review â€” merge only what you accept.",
    ])
    failed = failed_files or []
    remediation = "\n".join([
        "## Remediation changes",
        "",
        f"### Files remediated ({len(committed)})",
        "",
        "\n".join(f"- `{f}`" for f in committed),
        "",
        f"### Files without fixes ({len(failed)})",
        "",
        "\n".join(f"- `{f}`" for f in failed) or "_None_",
    ])
    details = _pr_remediation_details(fix_table, limit // 3)
    if details:
        remediation += "\n\n" + details

    sections = _split_report_sections(report or "")
    blocks = [f"{h}\n{sections[h].strip()}" for h in _PR_REPORT_SECTIONS if sections.get(h, "").strip()]
    report_md = "\n\n".join(blocks) if blocks else (report or "").strip()

    sep = "\n\n---\n\n"
    note = f"\n\n{report_note}" if report_note else ""
    budget = limit - len(head) - len(remediation) - 2 * len(sep) - len(note)
    parts = [head]
    if report_md and budget > 500:
        if len(report_md) > budget:
            cut = "\n\n*â€¦ report truncated â€” full report in the workflow run summary.*"
            report_md = report_md[: budget - len(cut)].rsplit("\n", 1)[0] + cut
        parts.append(report_md + note)
    parts.append(remediation)
    return sep.join(parts)[:limit]


# ===========================================================================
# Remediation PR creation
# ===========================================================================

def _normalize_github_repo_slug(repo: str) -> str:
    s = (repo or "").strip()
    s = re.sub(r"[\x00-\x1f\x7f]+", "", s)
    s = re.sub(r"\s+", "", s)
    s = s.strip("/")
    if s.lower().endswith(".git"):
        s = s[:-4]
    lower = s.lower()
    marker = "github.com/"
    idx = lower.find(marker)
    if idx != -1:
        s = s[idx + len(marker):]
    s = s.strip("/")
    parts = [p for p in s.split("/") if p]
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return s


def _normalize_git_ref(value: str) -> str:
    return re.sub(r"[\s\x00-\x1f\x7f]+", "", (value or "").strip())


class _GitHubClient:
    """Minimal GitHub REST client (this script's AzureDevOpsClient can't reach github.com)."""

    def __init__(self, token: str, base_url: str = "") -> None:
        self.token = token
        self.base_url = (base_url or os.environ.get("GITHUB_API_URL") or "https://api.github.com").rstrip("/")

    def _request(
        self, method: str, path: str, body: Optional[Dict[str, Any]] = None,
        *, expected_errors: Optional[set] = None,
    ) -> Any:
        url = f"{self.base_url}{path}" if path.startswith("/") else path
        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "UniFAI-ADO-Scanner/1.0",
        }
        data = json.dumps(body).encode() if body else None
        if data:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            err = exc.read().decode("utf-8", errors="replace")[:500]
            if expected_errors and exc.code in expected_errors:
                logger.debug("GitHub API %s %s â†’ %s (expected): %s", method, url, exc.code, err)
            else:
                logger.error("GitHub API %s %s â†’ %s: %s", method, url, exc.code, err)
            raise

    def create_branch(self, repo: str, branch_name: str, from_sha: str) -> None:
        repo = _normalize_github_repo_slug(repo)
        self._request("POST", f"/repos/{repo}/git/refs", {"ref": f"refs/heads/{branch_name}", "sha": from_sha})

    def get_file_blob_sha(self, repo: str, path: str, ref: str) -> Optional[str]:
        repo = _normalize_github_repo_slug(repo)
        encoded_path = urllib.parse.quote(path, safe="/")
        qref = urllib.parse.quote(ref, safe="")
        try:
            data = self._request(
                "GET", f"/repos/{repo}/contents/{encoded_path}?ref={qref}", expected_errors={404},
            )
            if isinstance(data, dict) and data.get("type") == "file" and data.get("sha"):
                return str(data["sha"])
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise
        return None

    def commit_file(
        self, repo: str, branch: str, path: str, content: bytes, message: str, sha: Optional[str] = None,
    ) -> str:
        repo = _normalize_github_repo_slug(repo)
        if sha is None:
            sha = self.get_file_blob_sha(repo, path, branch)
        payload: Dict[str, Any] = {
            "message": message, "content": base64.b64encode(content).decode(), "branch": branch,
        }
        if sha:
            payload["sha"] = sha
        encoded_path = urllib.parse.quote(path, safe="/")
        resp = self._request("PUT", f"/repos/{repo}/contents/{encoded_path}", payload)
        return resp["commit"]["sha"]

    def create_pull_request(self, repo: str, title: str, head: str, base: str, body: str) -> int:
        repo = _normalize_github_repo_slug(repo)
        resp = self._request("POST", f"/repos/{repo}/pulls", {
            "title": title, "head": head, "base": base, "body": body,
        })
        return resp["number"]


def _create_github_fix_pr(
    github_token: str,
    repo: str,
    branch: str,
    head_sha: str,
    validated_fixes: Dict[str, str],
    fix_table: List[Dict[str, str]],
    *,
    report: str = "",
    failed_files: Optional[List[str]] = None,
) -> Tuple[Optional[int], str]:
    """Commit LLM remediation (fix_code) patches to a branch and open a PR on github.com."""
    if not validated_fixes:
        return None, ""

    repo = _normalize_github_repo_slug(repo)
    branch = _normalize_git_ref(branch)
    head_sha = _normalize_git_ref(head_sha)
    if not repo:
        logger.error("Cannot create remediation PR: repo slug is empty after normalize.")
        return None, ""
    if not head_sha:
        logger.error("Cannot create remediation branch: head_sha is empty. Pass --head-sha.")
        return None, ""

    safe_branch = re.sub(r"[^a-zA-Z0-9._/-]", "-", branch)
    sha_short = head_sha[:7]
    timestamp = time.strftime("%m%d%H%M")
    remediation_branch = f"{REMEDIATION_BRANCH_PREFIX}-{safe_branch.replace('/', '-')}-{sha_short}-{timestamp}"

    scm = _GitHubClient(token=github_token)

    if len(head_sha) < 40:
        try:
            head_sha = scm._request("GET", f"/repos/{repo}/commits/{head_sha}")["sha"]
        except Exception as exc:
            logger.warning("Could not resolve short SHA %s: %s", head_sha, exc)

    try:
        logger.info("Creating remediation branch %s from %s", remediation_branch, sha_short)
        scm.create_branch(repo, remediation_branch, head_sha)
    except Exception as exc:
        logger.error("Failed to create/verify remediation branch: %s", exc)
        return None, remediation_branch

    committed: List[str] = []
    for filepath, content in sorted(validated_fixes.items()):
        blob_sha: Optional[str] = None
        try:
            blob_sha = scm.get_file_blob_sha(repo, filepath, head_sha)
        except Exception:
            pass
        policies = ", ".join({r["policy"] for r in fix_table if r.get("file") == filepath}) or "policy violations"
        message = f"fix({filepath}): remediate {policies} [unifai-ghp-scan]"
        try:
            scm.commit_file(repo, remediation_branch, filepath, content.encode("utf-8"), message, sha=blob_sha)
            committed.append(filepath)
            logger.info("Committed fix: %s", filepath)
        except Exception as exc:
            logger.error("Failed to commit %s: %s", filepath, exc)

    if not committed:
        logger.warning("No files committed â€” skipping PR creation")
        return None, remediation_branch

    title = f"[unifai-bot] fix: AI policy remediation for {branch}@{sha_short}"
    pr_body = _build_fix_pr_body(
        branch, sha_short, committed, failed_files, fix_table, report, GITHUB_PR_BODY_SAFE_LIMIT,
    )

    try:
        pr_number = scm.create_pull_request(repo, title, remediation_branch, branch, pr_body)
        logger.info("Created remediation PR #%d â€” https://github.com/%s/pull/%d", pr_number, repo, pr_number)
        return pr_number, remediation_branch
    except Exception as exc:
        logger.error("Failed to create remediation PR: %s", exc)
        return None, remediation_branch


def _create_fix_pr(
    scm: AzureDevOpsClient,
    branch: str,
    head_sha: str,
    validated_fixes: Dict[str, str],
    fix_table: List[Dict[str, str]],
    *,
    report: str = "",
    failed_files: Optional[List[str]] = None,
) -> Tuple[Optional[int], str]:
    """Commit fix_code patches to a remediation branch and open a PR."""
    if not validated_fixes:
        return None, ""

    safe_branch = re.sub(r"[^a-zA-Z0-9._/-]", "-", branch)
    sha_short = head_sha[:7]
    timestamp = time.strftime("%m%d%H%M")
    remediation_branch = f"{REMEDIATION_BRANCH_PREFIX}-{safe_branch.replace('/', '-')}-{sha_short}-{timestamp}"
    if len(head_sha) < 40:
        resolved: Optional[str] = None
        try:
            resolved = scm.get_branch_sha(branch)
        except Exception as exc:
            logger.warning("Could not resolve the tip of %s: %s", branch, exc)
        if not resolved:
            logger.error("Cannot create a remediation branch without the full commit id for %s", branch)
            return None, ""
        head_sha = resolved

    try:
        logger.info("Creating remediation branch %s from %s", remediation_branch, sha_short)
        scm.create_branch(remediation_branch, head_sha)
    except Exception as exc:
        logger.error("Failed to create remediation branch: %s", exc)
        return None, remediation_branch

    # changeType is per file: everything patched here was read off the checkout,
    # so "edit" is the norm, but ask rather than assume.
    existing: Dict[str, bool] = {}
    for filepath in validated_fixes:
        try:
            existing[filepath] = scm.file_exists(filepath, head_sha)
        except Exception as exc:
            logger.debug("Existence check failed for %s (%s) â€” treating as edit", filepath, exc)
            existing[filepath] = True

    committed = sorted(validated_fixes)
    policies = ", ".join(sorted({r["policy"] for r in fix_table if r.get("policy")})) or "policy violations"
    message = "\n".join(
        [f"fix: remediate {policies} [unifai-ado-scan]", ""]
        + [f"- {f}" for f in committed]
    )

    try:
        scm.push_commit(
            remediation_branch,
            head_sha,
            {f: validated_fixes[f].encode("utf-8") for f in committed},
            message,
            existing=existing,
        )
        logger.info("Committed %d fix(es) to %s", len(committed), remediation_branch)
    except Exception as exc:
        logger.error("Failed to push fixes to %s: %s", remediation_branch, exc)
        return None, remediation_branch

    title = f"[unifai-bot] fix: AI policy remediation for {branch}@{sha_short}"

    pr_body = _build_fix_pr_body(
        branch, sha_short, committed, failed_files, fix_table, report, AZURE_PR_DESCRIPTION_LIMIT,
        report_note="*Full scan report: see the `unifai-report` artifact on the pipeline run.*",
    )

    try:
        pr_id = scm.create_pull_request(title, remediation_branch, branch, pr_body)
        logger.info("Created remediation PR !%d â€” %s", pr_id, scm.pull_request_url(pr_id))
        return pr_id, remediation_branch
    except Exception as exc:
        logger.error("Failed to create remediation PR: %s", exc)
        return None, remediation_branch


# ===========================================================================
# Main scan orchestration
# ===========================================================================

def _azure_branch() -> str:
    """Short branch name from the agent's predefined variables.

    ``Build.SourceBranch`` is ``refs/heads/<name>`` on a CI run but
    ``refs/pull/<id>/merge`` on a PR validation run, where the branch actually
    under review is ``System.PullRequest.SourceBranch``. ``Build.SourceBranchName``
    is only the last path segment, so ``feature/a/b`` would come back as ``b`` â€”
    it is the last resort rather than the first choice.
    """
    for var in ("SYSTEM_PULLREQUEST_SOURCEBRANCH", "BUILD_SOURCEBRANCH"):
        raw = (os.environ.get(var) or "").strip()
        if raw.startswith("refs/heads/"):
            return raw[len("refs/heads/"):]
    return (os.environ.get("BUILD_SOURCEBRANCHNAME") or "").strip()


def _execute_scan(args: argparse.Namespace) -> int:
    org_url = args.org_url or os.environ.get("SYSTEM_TEAMFOUNDATIONCOLLECTIONURI", "")
    project = args.project or os.environ.get("SYSTEM_TEAMPROJECT", "")
    repo_name = args.repo or os.environ.get("BUILD_REPOSITORY_NAME", "")
    repo_id = args.repo_id or os.environ.get("BUILD_REPOSITORY_ID", "") or repo_name
    branch = args.branch or _azure_branch()
    head_sha = args.head_sha or os.environ.get("BUILD_SOURCEVERSION", "")
    source_path = os.path.abspath(args.source_path)
    server_url = args.mcp_server_url or os.environ.get("MCP_SERVER_URL", "") or MCP_SERVER_URL

    repo = f"{project}/{repo_name}" if project and repo_name else (repo_name or project)
    source_code_repo = os.environ.get("BUILD_REPOSITORY_URI", "")
    if not source_code_repo:
        if org_url and project and repo_name:
            source_code_repo = (
                f"{org_url.rstrip('/')}/{urllib.parse.quote(project, safe='')}"
                f"/_git/{urllib.parse.quote(repo_name, safe='')}"
            )
        else:
            source_code_repo = source_path

    # Validate config
    missing = [n for n, v in [
        ("BUILD_REPOSITORY_NAME / --repo", repo_name),
        ("BUILD_SOURCEBRANCH / --branch", branch),
    ] if not v]
    if missing:
        output = build_json_output(
            status="error", repo=repo, branch=branch, head_sha=head_sha,
            source_code_repo=source_code_repo, files_scanned=0, batches=0, failed_batches=0,
            violations=[], scan_errors=[f"Missing required config: {', '.join(missing)}"],
        )
        print_human_output(output)
        return 2

    try:
        bearer_getter = build_bearer_getter()
        # Eagerly fetch a token at startup to catch auth errors early
        bearer_getter()
        logger.info("Auth OK â€” LINEAJE_PAT_TOKEN accepted")
    except Exception as exc:
        output = build_json_output(
            status="error", repo=repo, branch=branch, head_sha=head_sha,
            source_code_repo=source_code_repo, files_scanned=0, batches=0, failed_batches=0,
            violations=[], scan_errors=[f"Auth failed: {exc}"],
        )
        print_human_output(output)
        return 2

    run_id = time.strftime("%Y%m%d_%H%M%S")
    scan_start = time.perf_counter()

    logger.info("Scanning source path: %s (repo=%s branch=%s sha=%s)", source_path, repo, branch, head_sha[:7] if head_sha else "?")

    # Step 1: Collect files
    file_list = collect_repo_files(source_path, getattr(args, "exclude", None))
    if not file_list:
        logger.info("No scannable files found")
        output = build_json_output(
            status="compliant", repo=repo, branch=branch, head_sha=head_sha,
            source_code_repo=source_code_repo, files_scanned=0, batches=0, failed_batches=0,
            violations=[],
        )
        print_human_output(output)
        return 0

    batch_size = _batch_size(len(file_list))
    batches, code_files, manifest_files = pin_manifests_to_first_batch(file_list, batch_size)
    logger.info(
        "Files: %d total (%d code, %d manifests in batch 1) â†’ %d batch(es)",
        len(file_list), len(code_files), len(manifest_files), len(batches),
    )

    # Step 2: MCP scan
    with tempfile.TemporaryDirectory(prefix="ado-repo-scan-") as temp_dir:
        (
            all_violations, all_remediation_actions, all_reports, all_aibom,
            failed_batches_count, failure_details, all_stub_insertions, uploaded,
        ) = parallel_batch_scan(
            batches=batches,
            source_dir=source_path,
            temp_dir=temp_dir,
            source_code_repo=source_code_repo,
            branch=branch,
            head_sha=head_sha,
            run_id=run_id,
            server_url=server_url,
            bearer_getter=bearer_getter,
            manifest_files=manifest_files or None,
        )

    elapsed = time.perf_counter() - scan_start
    logger.info(
        "Scan complete in %.1fs: %d violation(s), %d AIBOM entr(ies), %d failed batch(es)",
        elapsed, len(all_violations), len(all_aibom), failed_batches_count,
    )

    combined_report = _merge_batch_reports(
        all_reports,
        total_violations=len(all_violations),
        total_remediation_actions=len(all_remediation_actions),
        total_elapsed=elapsed,
    )

    if failed_batches_count and not all_violations:
        output = build_json_output(
            status="error", repo=repo, branch=branch, head_sha=head_sha,
            source_code_repo=source_code_repo, files_scanned=len(file_list),
            batches=len(batches), failed_batches=failed_batches_count,
            violations=[], aibom=all_aibom, report=combined_report,
            scan_errors=failure_details,
        )
        print_human_output(output)
        return 1

    status = "compliant" if not all_violations else "violations_found"
    if failed_batches_count:
        status = "error"

    # Step 3: LLM remediation â€” apply the fix_code patches from the server's
    # remediation_actions to the checkout (guardrail stub insertion is disabled on the
    # server and never applied here), and add the remediation to the Lineaje report.
    remediation_pr_number: Optional[int] = None
    remediation_branch = ""
    remediation_pr_url = ""
    failed_rem_files: List[str] = []
    validated_fixes: Dict[str, str] = {}
    fix_table: List[Dict[str, str]] = []

    if all_stub_insertions:
        logger.info(
            "STEP 3: ignoring %d guardrail stub insertion(s) â€” stubs are disabled; using LLM remediation",
            len(all_stub_insertions),
        )
    if all_remediation_actions:
        logger.info("STEP 3: applying LLM remediation (%d action(s))", len(all_remediation_actions))
        validated_fixes, failed_rem_files, fix_table = apply_pipeline_fix_code_to_clone(
            all_remediation_actions, source_path, file_list,
        )
        logger.info(
            "STEP 3: %d file(s) patched, %d file(s) need a manual fix",
            len(validated_fixes), len(failed_rem_files),
        )

    # GitHub token takes precedence over Azure DevOps config.
    github_token = _normalize_token(
        getattr(args, "github_token", None)
        or os.environ.get("GH_TOKEN", "")
        or os.environ.get("GITHUB_TOKEN", "")
    )
    azure_token = _normalize_token(
        getattr(args, "azure_token", None)
        or os.environ.get("SYSTEM_ACCESSTOKEN", "")
        or os.environ.get("AZURE_DEVOPS_EXT_PAT", "")
        or os.environ.get("AZURE_DEVOPS_PAT", "")
    )
    use_github = bool(github_token)
    should_create_pr = bool((github_token or azure_token) and getattr(args, "create_fix_pr", False))
    missing_scm: List[str] = []
    if should_create_pr and not use_github:
        missing_scm = [n for n, v in [
            ("a token (SYSTEM_ACCESSTOKEN / --azure-token)", azure_token),
            ("SYSTEM_TEAMFOUNDATIONCOLLECTIONURI / --org-url", org_url),
            ("SYSTEM_TEAMPROJECT / --project", project),
        ] if not v]

    if should_create_pr and use_github:
        if validated_fixes:
            logger.info("Creating remediation PR on GitHub (%d file(s))", len(validated_fixes))
            try:
                remediation_pr_number, remediation_branch = _create_github_fix_pr(
                    github_token, repo, branch, head_sha,
                    validated_fixes, fix_table,
                    report=combined_report, failed_files=failed_rem_files,
                )
                if remediation_pr_number:
                    remediation_pr_url = (
                        f"https://github.com/{_normalize_github_repo_slug(repo)}"
                        f"/pull/{remediation_pr_number}"
                    )
            except Exception as exc:
                logger.error("Remediation PR failed: %s", exc)
        else:
            logger.info("No LLM remediation patches to commit for a remediation PR")
    elif should_create_pr and missing_scm:
        logger.info("Skipping remediation PR â€” missing %s", ", ".join(missing_scm))
    elif should_create_pr and validated_fixes:
        logger.info("Creating remediation PR (%d file(s))", len(validated_fixes))
        try:
            remediation_pr_number, remediation_branch = _create_fix_pr(
                AzureDevOpsClient(
                    token=azure_token,
                    org_url=org_url,
                    project=project,
                    repository=repo_id,
                ),
                branch, head_sha,
                validated_fixes, fix_table,
                report=combined_report, failed_files=failed_rem_files,
            )
            if remediation_pr_number:
                remediation_pr_url = AzureDevOpsClient(
                    token=azure_token, org_url=org_url, project=project, repository=repo_id,
                ).pull_request_url(remediation_pr_number)
        except Exception as exc:
            logger.error("Remediation PR failed: %s", exc)
    elif should_create_pr:
        logger.info("No LLM remediation patches to commit for a remediation PR")

    output = build_json_output(
        status=status, repo=repo, branch=branch, head_sha=head_sha,
        source_code_repo=source_code_repo, files_scanned=len(file_list),
        batches=len(batches), failed_batches=failed_batches_count,
        violations=all_violations, aibom=all_aibom, report=combined_report,
        remediation_pr=remediation_pr_number,
        remediation_branch=remediation_branch,
        remediation_pr_url=remediation_pr_url,
        failed_remediation_files=failed_rem_files,
        scan_errors=failure_details,
        remediation_actions=all_remediation_actions,
        uploaded_entities=uploaded["entities"],
        uploaded_findings=uploaded["findings"],
        source_archive_s3_uploaded=uploaded["source_archive_s3_uploaded"],
    )
    print_human_output(output)
    return 0

# ===========================================================================
# CLI
# ===========================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lineaje AI Policy Scanner â€” Azure Pipelines edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source-path", default=".",
        help="Path to the checked-out source code (default: current directory)",
    )
    parser.add_argument(
        "--exclude", default=[], action="append", metavar="PATH",
        help="Path relative to --source-path to keep out of the scan, repeatable "
             "(the pipeline passes --exclude .git --exclude .azuredevops)",
    )
    parser.add_argument(
        "--org-url", default="",
        help="Azure DevOps collection URL, e.g. https://dev.azure.com/myorg/ "
             "(default: $SYSTEM_TEAMFOUNDATIONCOLLECTIONURI)",
    )
    parser.add_argument(
        "--project", default="",
        help="Azure DevOps project name (default: $SYSTEM_TEAMPROJECT)",
    )
    parser.add_argument(
        "--repo", default="",
        help="Repository name (default: $BUILD_REPOSITORY_NAME)",
    )
    parser.add_argument(
        "--repo-id", default="",
        help="Repository GUID used for REST calls (default: $BUILD_REPOSITORY_ID, "
             "falling back to the repository name)",
    )
    parser.add_argument(
        "--branch", default="",
        help="Branch name without the refs/heads/ prefix (default: derived from "
             "$BUILD_SOURCEBRANCH, or $SYSTEM_PULLREQUEST_SOURCEBRANCH on a PR run)",
    )
    parser.add_argument(
        "--head-sha", default="",
        help="Commit id (default: $BUILD_SOURCEVERSION)",
    )
    parser.add_argument(
        "--mcp-server-url", default="",
        help=f"MCP server URL (default: {MCP_SERVER_URL})",
    )
    parser.add_argument(
        "--azure-token", default="",
        help="Azure DevOps token for creating remediation PRs (default: "
             "$SYSTEM_ACCESSTOKEN, then $AZURE_DEVOPS_EXT_PAT, then $AZURE_DEVOPS_PAT). "
             "If not set, violations are reported but no PR is created.",
    )
    parser.add_argument(
        "--github-token", default="",
        help="GitHub token for creating remediation PRs when --repo is a GitHub "
             "repo (default: $GH_TOKEN, then $GITHUB_TOKEN). Takes precedence over "
             "--azure-token/--org-url/--project when set â€” this script talks to "
             "whichever host the token is for, GitHub or Azure DevOps, not both.",
    )
    parser.add_argument(
        "--create-fix-pr", default=False, action="store_true",
        help="Create a remediation PR with fix_code patches (default: false).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG logging to stderr",
    )
    return parser.parse_args(argv or sys.argv[1:])


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )
    # Always show INFO from this logger regardless of --debug
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    try:
        return _execute_scan(args)
    except Exception:
        logger.exception("Unhandled error")
        err = {"status": "error", "scan_errors": ["Unhandled exception â€” see stderr logs"]}
        print_human_output(err)
        return 1


if __name__ == "__main__":
    sys.exit(main())
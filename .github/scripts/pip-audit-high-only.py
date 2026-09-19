#!/usr/bin/env python3
"""Fail only when pip-audit JSON contains HIGH or CRITICAL OSV advisories."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

FAIL_SEVERITIES = frozenset({"HIGH", "CRITICAL"})
OSV_URL = "https://api.osv.dev/v1/vulns/{vuln_id}"


def osv_severity(vuln_id: str, cache: dict[str, str]) -> str:
    if vuln_id in cache:
        return cache[vuln_id]
    severity = "UNKNOWN"
    try:
        with urllib.request.urlopen(
            OSV_URL.format(vuln_id=vuln_id),
            timeout=30,
        ) as response:
            payload = json.loads(response.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        cache[vuln_id] = severity
        return severity

    db = payload.get("database_specific") or {}
    if isinstance(db.get("severity"), str):
        severity = db["severity"].upper()
    cache[vuln_id] = severity
    return severity


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "pip-audit.json"
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)

    cache: dict[str, str] = {}
    blocking: list[tuple[str, str, str, list[str]]] = []
    seen: set[tuple[str, str]] = set()

    for dep in report.get("dependencies", []):
        name = dep.get("name", "?")
        for vuln in dep.get("vulns", []):
            vuln_id = vuln.get("id")
            if not vuln_id:
                continue
            key = (name, vuln_id)
            if key in seen:
                continue
            seen.add(key)
            severity = osv_severity(vuln_id, cache)
            if severity in FAIL_SEVERITIES:
                blocking.append(
                    (name, vuln_id, severity, vuln.get("fix_versions") or []),
                )

    if blocking:
        print("pip-audit: HIGH/CRITICAL vulnerabilities (CI policy):")
        for name, vuln_id, severity, fixes in blocking:
            fix = ", ".join(fixes) if fixes else "none listed"
            print(f"  {name}: {vuln_id} [{severity}] fix: {fix}")
        return 1

    print("pip-audit: no HIGH/CRITICAL vulnerabilities")
    return 0


if __name__ == "__main__":
    sys.exit(main())

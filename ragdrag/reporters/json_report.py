"""JSON report generator for RAGdrag findings.

Produces machine-readable output compatible with security tooling pipelines.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import click

from ragdrag import __version__
from ragdrag.core.fingerprint import FingerprintResult

# Confidence level colors
_CONFIDENCE_COLORS = {
    "high": "red",
    "medium": "yellow",
    "low": "green",
}


def generate_report(
    fingerprint_result: FingerprintResult,
    output_path: str | Path | None = None,
) -> dict:
    """Generate a JSON report from fingerprint results.

    Args:
        fingerprint_result: Results from a fingerprint scan.
        output_path: Optional file path to write the report. If None,
            returns the dict without writing.

    Returns:
        The report as a dictionary.
    """
    report = {
        "tool": "ragdrag",
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": fingerprint_result.target,
        "summary": {
            "rag_detected": fingerprint_result.rag_detected,
            "vector_db": fingerprint_result.vector_db,
            "detected_response_field": fingerprint_result.detected_response_field,
            "total_findings": len(fingerprint_result.findings),
            "high_confidence": sum(
                1 for f in fingerprint_result.findings if f.confidence == "high"
            ),
            "medium_confidence": sum(
                1 for f in fingerprint_result.findings if f.confidence == "medium"
            ),
            "low_confidence": sum(
                1 for f in fingerprint_result.findings if f.confidence == "low"
            ),
        },
        "infrastructure": fingerprint_result.infrastructure,
        "findings": [
            {
                "technique_id": f.technique_id,
                "technique_name": f.technique_name,
                "confidence": f.confidence,
                "detail": f.detail,
                "evidence": f.evidence,
            }
            for f in fingerprint_result.findings
        ],
        "timing": fingerprint_result.timing_stats,
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")

    return report


def _build_summary_box(report: dict) -> str:
    """Build a bordered summary box for CLI output.

    Returns plain text (no ANSI) so callers can apply click.style() or
    tests can assert on content.
    """
    summary = report["summary"]
    h = summary["high_confidence"]
    m = summary["medium_confidence"]
    lo = summary["low_confidence"]
    total = summary["total_findings"]

    target = report["target"]
    rag = str(summary["rag_detected"])
    vdb = summary["vector_db"] or "unknown"
    findings_str = f"{total} ({h}H / {m}M / {lo}L)"
    resp_field = summary.get("detected_response_field")

    infra = report.get("infrastructure", {})
    server_tech = infra.get("server_tech", [])
    frameworks = [f["name"] for f in infra.get("rag_frameworks", [])]

    rows = [
        f"  Target:       {target}",
        f"  RAG Detected: {rag}",
        f"  Vector DB:    {vdb}",
    ]
    if resp_field:
        rows.append(f"  Response Key:  {resp_field}")
    if server_tech:
        rows.append(f"  Server:        {', '.join(server_tech)}")
    if frameworks:
        rows.append(f"  Framework:     {', '.join(frameworks)}")
    rows.append(f"  Findings:     {findings_str}")
    inner_width = max(len(r) for r in rows) + 2
    top = "\u2554" + "\u2550" * inner_width + "\u2557"
    bot = "\u255a" + "\u2550" * inner_width + "\u255d"
    mid_lines = [
        "\u2551" + r.ljust(inner_width) + "\u2551"
        for r in rows
    ]
    return "\n".join([top, *mid_lines, bot])


def format_summary(report: dict, *, color: bool = True) -> str:
    """Format a report summary for CLI output.

    Args:
        report: Report dictionary from generate_report.
        color: Whether to apply ANSI color codes via click.style().

    Returns:
        Human-readable summary string.
    """
    box = _build_summary_box(report)
    lines = [box, ""]

    for f in report["findings"]:
        conf = f["confidence"]
        tag = conf.upper()
        conf_color = _CONFIDENCE_COLORS.get(conf, "white")
        if color:
            label = click.style(f"[{tag}]", fg=conf_color, bold=True)
        else:
            label = f"[{tag}]"
        lines.append(f"  [!] {label} {f['technique_id']}: {f['technique_name']}")
        lines.append(f"      {f['detail']}")
        lines.append("")

    return "\n".join(lines)

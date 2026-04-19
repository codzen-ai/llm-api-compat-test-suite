from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from config import ProviderConfig
    from model_profile import ResolvedModel


def _models_section(
    provider: ProviderConfig,
    resolved_models: list[ResolvedModel] | None,
) -> list[str]:
    """Render the per-model profile detail table.

    Each configured model gets one row listing the profile path, snapshot,
    created_at, and whether the snapshot was pinned in config or auto-picked
    as the latest by the registry. The point of the table is that a reader
    can reconstruct *exactly* which ground-truth was used for a run.
    """
    lines = [
        "## Models",
        "",
        "| Model | Profile | Snapshot | Created | Resolution |",
        "|-------|---------|----------|---------|------------|",
    ]
    by_name: dict[str, ResolvedModel] = {}
    if resolved_models:
        by_name = {rm.name: rm for rm in resolved_models}

    for m in provider.models:
        rm = by_name.get(m.name)
        if rm is None or rm.profile is None:
            # Transitional path: no profile attached (TODO 7 removes this).
            lines.append(
                f"| {m.name} | _none — using config.capabilities_ "
                "| — | — | fallback |"
            )
            continue
        p = rm.profile
        path_str = "—"
        if p.source_path is not None:
            try:
                path_str = str(p.source_path.relative_to(Path.cwd()))
            except ValueError:
                path_str = str(p.source_path)
        resolution = "pinned" if m.profile_snapshot else "auto-latest"
        lines.append(
            f"| {m.name} | `{path_str}` | {p.snapshot} "
            f"| {p.created_at.isoformat()} | {resolution} |"
        )
    lines.append("")
    return lines


class TestResult(BaseModel):
    node_id: str
    outcome: str  # "passed", "failed", "skipped", "error"
    duration: float
    log_file: Path | None = None
    failure_message: str = ""


class ReportCollector(BaseModel):
    results: list[TestResult] = Field(default_factory=lambda: list[TestResult]())
    report_dir: Path = Field(default_factory=lambda: Path("reports"))

    def add_result(self, result: TestResult) -> None:
        self.results.append(result)

    def generate_summary(
        self,
        provider: ProviderConfig | None = None,
        resolved_models: list[ResolvedModel] | None = None,
    ) -> Path:
        summary_path = self.report_dir / "summary.md"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        passed = [r for r in self.results if r.outcome == "passed"]
        failed = [r for r in self.results if r.outcome == "failed"]
        skipped = [r for r in self.results if r.outcome == "skipped"]
        errored = [r for r in self.results if r.outcome == "error"]

        lines = [
            "# LLM API Compatibility Test Report",
            "",
        ]

        if provider:
            lines.extend([
                "## Configuration",
                "",
                "| Key | Value |",
                "|-----|-------|",
                f"| Provider | {provider.name} |",
                f"| Base URL | `{provider.base_url}` |",
                f"| API Format | {provider.api_format} |",
                f"| Auth Type | {provider.auth_type or 'auto'} |",
                f"| Verify SSL | {provider.verify_ssl} |",
                "",
            ])
            lines.extend(_models_section(provider, resolved_models))

        lines.extend([
            "## Summary",
            "",
            "| Status | Count |",
            "|--------|-------|",
            f"| Passed | {len(passed)} |",
            f"| Failed | {len(failed)} |",
            f"| Skipped | {len(skipped)} |",
            f"| Error | {len(errored)} |",
            f"| **Total** | **{len(self.results)}** |",
            "",
        ])

        # Group by test directory (api format)
        groups: dict[str, list[TestResult]] = {}
        for r in self.results:
            parts = r.node_id.split("/")
            group = parts[1] if len(parts) > 1 else "unknown"
            groups.setdefault(group, []).append(r)

        for group_name, group_results in sorted(groups.items()):
            group_passed = sum(
                1 for r in group_results if r.outcome == "passed"
            )
            group_total = len(group_results)
            lines.extend([
                f"## {group_name} ({group_passed}/{group_total} passed)",
                "",
                "| Test | Status | Duration | Log |",
                "|------|--------|----------|-----|",
            ])
            for r in group_results:
                status_icon = {
                    "passed": "PASS",
                    "failed": "FAIL",
                    "skipped": "SKIP",
                    "error": "ERROR",
                }.get(r.outcome, r.outcome)

                log_link = ""
                if r.log_file and r.log_file.exists():
                    rel = r.log_file.relative_to(self.report_dir)
                    log_link = f"[log]({rel})"

                test_name = (
                    r.node_id.split("::")[-1]
                    if "::" in r.node_id
                    else r.node_id
                )
                lines.append(
                    f"| {test_name} | {status_icon}"
                    f" | {r.duration:.2f}s | {log_link} |"
                )

            lines.append("")

        if failed:
            lines.extend([
                "## Failure Details",
                "",
            ])
            for r in failed:
                lines.extend([
                    f"### {r.node_id}",
                    "",
                    "```",
                    r.failure_message,
                    "```",
                    "",
                ])

        summary_path.write_text("\n".join(lines), encoding="utf-8")
        return summary_path

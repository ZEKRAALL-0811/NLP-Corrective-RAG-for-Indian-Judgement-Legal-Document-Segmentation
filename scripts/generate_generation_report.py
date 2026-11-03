"""
Generate artifacts/reports/generation_report.md from artifacts/logs/generation_results.jsonl.
Contains:
1. Status counts and breakdown
2. Latency table (per-node and total wall time)
3. Full table of all 30 evaluated rows (id, sample, question, gold markers, status, citations, ungrounded_numbers, answer)
"""

import json
from pathlib import Path

def generate_report():
    jsonl_path = Path("artifacts/logs/generation_results.jsonl")
    report_path = Path("artifacts/reports/generation_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} does not exist.")
        return

    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line.strip()))

    total_rows = len(rows)
    gold_rows = [r for r in rows if r.get("id", "").startswith("gold_")]
    unans_rows = [r for r in rows if r.get("id", "").startswith("unans_")]

    # 1. Status Counts
    status_counts = {}
    for r in rows:
        st = r.get("final_status", "unknown")
        status_counts[st] = status_counts.get(st, 0) + 1

    gold_status_counts = {}
    for r in gold_rows:
        st = r.get("final_status", "unknown")
        gold_status_counts[st] = gold_status_counts.get(st, 0) + 1

    unans_status_counts = {}
    for r in unans_rows:
        st = r.get("final_status", "unknown")
        unans_status_counts[st] = unans_status_counts.get(st, 0) + 1

    # 2. Latencies
    node_times = {}
    total_times = []
    for r in rows:
        pnl = r.get("per_node_latency", {})
        total_q_ms = sum(pnl.values())
        total_times.append(total_q_ms)
        for node, ms in pnl.items():
            if node not in node_times:
                node_times[node] = []
            node_times[node].append(ms)

    lines = []
    lines.append("# Module 9 Generation Report")
    lines.append("")
    lines.append(f"**Total Questions Evaluated**: {total_rows} ({len(gold_rows)} Answerable Gold, {len(unans_rows)} Unanswerable)")
    lines.append("")
    lines.append("## 1. Status Breakdown")
    lines.append("")
    lines.append("| Status | Overall Count | Gold (22) | Unanswerable (8) |")
    lines.append("|---|---|---|---|")
    all_statuses = sorted(set(status_counts.keys()))
    for st in all_statuses:
        lines.append(f"| `{st}` | {status_counts.get(st, 0)} | {gold_status_counts.get(st, 0)} | {unans_status_counts.get(st, 0)} |")
    lines.append("")

    lines.append("## 2. Latency Table (Per-Node and Total in ms)")
    lines.append("")
    lines.append("| Node | Invocations | Mean Latency (ms) | Min (ms) | Max (ms) |")
    lines.append("|---|---|---|---|---|")
    for node in sorted(node_times.keys()):
        times = node_times[node]
        mean_ms = sum(times) / len(times)
        min_ms = min(times)
        max_ms = max(times)
        lines.append(f"| `{node}` | {len(times)} | {mean_ms:.1f} | {min_ms:.1f} | {max_ms:.1f} |")
    if total_times:
        lines.append(f"| **Total Pipeline Wall Time** | {len(total_times)} | {sum(total_times)/len(total_times):.1f} | {min(total_times):.1f} | {max(total_times):.1f} |")
    lines.append("")

    lines.append("## 3. Evaluated Questions (All 30 Rows)")
    lines.append("")
    lines.append("| ID | Sample | Question | Gold Role / Markers | Final Status | Citations | Ungrounded? | Answer Snippet |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        rid = r.get("id", "")
        sample = r.get("sample", "")
        q = r.get("question", "").replace("|", "\\|")
        markers = ", ".join(r.get("gold_markers", [])) or "None"
        role = r.get("gold_role", "")
        role_marker = f"{role} [{markers}]"
        status = r.get("final_status", "")
        cits = str(r.get("citations", []))
        ungrounded = str(r.get("ungrounded_numbers", False))
        ans = r.get("answer", "").replace("\n", " ").replace("|", "\\|")
        ans_snip = (ans[:90] + "...") if len(ans) > 90 else ans
        lines.append(f"| `{rid}` | {sample} | {q} | {role_marker} | `{status}` | {cits} | {ungrounded} | {ans_snip} |")

    report_content = "\n".join(lines)
    report_path.write_text(report_content, encoding="utf-8")
    print(f"Generated {report_path} successfully.")

if __name__ == "__main__":
    generate_report()

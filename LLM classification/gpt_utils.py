import ast
from collections.abc import Iterable


def parse_list(x):

    if isinstance(x, str):
        s = x.strip()
        if s.startswith('[') and s.endswith(']'):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [p.strip() for p in parsed if isinstance(p, str)]
            except Exception:
                inner = s[1:-1]
                return [p.strip() for p in inner.split(',') if p.strip()]
        return [s.strip()]
        # if it's already a list of labels
    if isinstance(x, list):
        return [p.strip() for p in x if isinstance(p, str)]
    return [str(x)]

import os, json, re
from datetime import datetime

def save_eval(metrics, report_txt, out_dir = "eval_logs", run_name = None):

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"\W+", "_", run_name.strip()) if run_name else "run"

    base = os.path.join(out_dir, f"{ts}_{slug}")
    txt_path  = base + ".txt"

    # write TXT (metrics + report)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Evaluation Metrics ===\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== Classification Report ===\n")
        f.write(report_txt if report_txt.endswith("\n") else report_txt + "\n")

    print(f"[Saved] {txt_path}")
    return {"txt": txt_path}

r"""
run_jobs.py — run every job bundle in a folder through staged_reintegrate.py.

Drop the bundles exported from the mask tool on the project page (.zip files,
or unzipped folders holding a job.json; see JOB_FORMAT.md) into one folder and
run, from the repo root:

    python pipeline/run_jobs.py jobs/
    python pipeline/run_jobs.py jobs/ --out results --force -- --region_variants 8

Each job runs `staged_reintegrate.py --job` in its own subprocess, so one
failure doesn't stop the rest; a summary prints at the end. Outputs go to
<out>/<job name>/ (default out = <folder>/out). A finished job gets a DONE
file there and is skipped next time unless --force. Defaults are the paper's
settled recipe (--model flux_base --lora_variant trigonly_v2); on a smaller
GPU use --model sdxl_base --lora_variant v1. Anything after `--` is passed on
to staged_reintegrate.py and wins over the jobs' own settings.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def job_name(bundle: Path) -> str:
    """The job's `name` (its output folder), read without extracting the zip."""
    if bundle.is_dir():
        return json.loads((bundle / "job.json").read_text(encoding="utf-8"))["name"]
    with zipfile.ZipFile(bundle) as z:
        js = [n for n in z.namelist() if n == "job.json" or n.endswith("/job.json")]
        return json.loads(z.read(min(js, key=len)).decode("utf-8"))["name"]


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description="Run every job bundle in a folder.")
    ap.add_argument("folder", help="folder of job bundles (*.zip and/or folders with job.json)")
    ap.add_argument("--out", default=None, help="output root (default <folder>/out)")
    ap.add_argument("--force", action="store_true", help="rerun jobs that already have a DONE marker")
    ap.add_argument("--model", default="flux_base")
    ap.add_argument("--lora_variant", default="trigonly_v2")
    args = ap.parse_args(argv)

    folder = Path(args.folder)
    out = Path(args.out) if args.out else folder / "out"
    bundles = sorted(folder.glob("*.zip")) + \
        sorted(d for d in folder.iterdir() if d.is_dir() and (d / "job.json").exists()
               and d.resolve() != out.resolve() and not (d / "DONE").exists())
    if not bundles:
        raise SystemExit(f"no job bundles (*.zip or folders with job.json) in {folder}")

    script = Path(__file__).resolve().parent / "staged_reintegrate.py"
    done, skipped, failed, seen = [], [], [], {}
    for i, b in enumerate(bundles, 1):
        try:
            name = job_name(b)
        except Exception as e:                           # unreadable bundle: report, carry on
            failed.append((b.name, f"can't read job.json ({e})"))
            continue
        if name in seen:
            failed.append((b.name, f"same job name '{name}' as {seen[name]}"))
            continue
        seen[name] = b.name
        marker = out / name / "DONE"
        if marker.exists() and not args.force:
            skipped.append(b.name)
            print(f"[{i}/{len(bundles)}] {b.name}: already done ({marker}), skipping")
            continue
        cmd = [sys.executable, str(script), "--job", str(b), "--model", args.model,
               "--lora_variant", args.lora_variant, "--out", str(out)] + extra
        print(f"\n[{i}/{len(bundles)}] {b.name} -> {out / name}\n  " + " ".join(cmd), flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n{' '.join(cmd)}\n")
            done.append(b.name)
        else:
            failed.append((b.name, f"exit code {rc}"))
        print(f"  {'ok' if rc == 0 else 'FAILED'} in {time.time() - t0:.0f}s", flush=True)

    print(f"\n=== {len(done)} done, {len(skipped)} skipped (already done), {len(failed)} failed ===")
    for n in done:
        print(f"  done     {n}")
    for n in skipped:
        print(f"  skipped  {n}")
    for n, why in failed:
        print(f"  FAILED   {n}: {why}")
    print(f"outputs -> {out}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

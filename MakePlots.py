# Q3: Scripted generation of required plots from W&B runs (besides the W&B UI).
# Produces PNGs in ./plots/ so you can drop them into the PDF.

import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

def fetch_runs_df(entity, project, top_k=None):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    rows = []
    for r in runs:
        summ = dict(r.summary) if r.summary else {}
        cfg  = dict(r.config) if r.config else {}
        val_acc = summ.get("val_acc", None)
        if val_acc is None:
            continue
        rows.append({
            "run_id": r.id,
            "activation": cfg.get("activation"),
            "optimizer":  cfg.get("optimizer"),
            "batch_size": cfg.get("batch_size"),
            "epochs":     cfg.get("epochs"),
            "lr":         cfg.get("lr"),
            "weight_decay": cfg.get("weight_decay"),
            "val_acc":    val_acc
        })
    df = pd.DataFrame(rows).dropna()
    df = df.sort_values("val_acc", ascending=False)
    if top_k: df = df.head(top_k)
    return df

def make_parallel_coordinates(df, outpath):
    # Map categorical to codes for plotting
    plot_cols = ["activation","optimizer","batch_size","epochs","lr","weight_decay","val_acc"]
    dfp = df[plot_cols].copy()

    cat_maps = {}
    for col in ["activation","optimizer"]:
        dfp[col] = dfp[col].astype("category")
        cat_maps[col] = dict(enumerate(dfp[col].cat.categories))
        dfp[col] = dfp[col].cat.codes

    # Normalize numeric columns to [0,1]
    def norm(col):
        c = dfp[col].astype(float)
        mn, mx = c.min(), c.max()
        return (c - mn) / (mx - mn + 1e-12)
    numeric_cols = ["batch_size","epochs","lr","weight_decay","val_acc"]
    for col in numeric_cols:
        dfp[col] = norm(col)

    x = np.arange(len(plot_cols))
    plt.figure(figsize=(12,6))
    for i in range(len(dfp)):
        y = dfp.iloc[i].values.astype(float)
        plt.plot(x, y, alpha=0.35)
    plt.xticks(x, plot_cols, rotation=20)
    plt.title("Parallel Coordinates (normalized) — colored by line index")
    plt.grid(True, axis="y", alpha=0.3)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def make_run_plots(run, outdir):
    hist = run.history(keys=["_step","val_acc","train_acc","train_loss","val_loss"], pandas=True)
    os.makedirs(outdir, exist_ok=True)

    # Scatter: val_acc vs step
    plt.figure(figsize=(7,5))
    plt.scatter(hist["_step"], hist["val_acc"], s=10)
    plt.xlabel("Step"); plt.ylabel("Validation Accuracy")
    plt.title(f"Validation Accuracy vs Step — run {run.id}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "val_acc_vs_step.png"), dpi=150); plt.close()

    # Curves: loss
    plt.figure(figsize=(7,5))
    plt.plot(hist["_step"], hist["train_loss"], label="train_loss")
    plt.plot(hist["_step"], hist["val_loss"],   label="val_loss")
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.legend()
    plt.title(f"Loss Curves — run {run.id}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "curves_loss.png"), dpi=150); plt.close()

    # Curves: accuracy
    plt.figure(figsize=(7,5))
    plt.plot(hist["_step"], hist["train_acc"], label="train_acc")
    plt.plot(hist["_step"], hist["val_acc"],   label="val_acc")
    plt.xlabel("Step"); plt.ylabel("Accuracy"); plt.legend()
    plt.title(f"Accuracy Curves — run {run.id}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "curves_acc.png"), dpi=150); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True, help="W&B entity (username or team)")
    ap.add_argument("--project", default="cs6886-vgg6-cifar10")
    ap.add_argument("--baseline_run", default=None, help="Optional run id to plot; otherwise best val_acc run")
    ap.add_argument("--top_k", type=int, default=50, help="Top-K runs to include in parallel-coordinates")
    ap.add_argument("--outdir", default="plots")
    args = ap.parse_args()

    api = wandb.Api()
    df = fetch_runs_df(args.entity, args.project, top_k=args.top_k)
    if df.empty:
        print("No runs found with val_acc.")
        return

    # Parallel coordinates across runs
    make_parallel_coordinates(df, os.path.join(args.outdir, "parallel_coordinates.png"))
    df.to_csv(os.path.join(args.outdir, "runs_summary.csv"), index=False)

    # Pick run (best or specified)
    project_path = f"{args.entity}/{args.project}"
    if args.baseline_run:
        run = api.run(f"{project_path}/{args.baseline_run}")
    else:
        best_id = df.iloc[0]["run_id"]
        run = api.run(f"{project_path}/{best_id}")

    make_run_plots(run, args.outdir)
    print(f"Saved plots to: {args.outdir}")

if __name__ == "__main__":
    main()

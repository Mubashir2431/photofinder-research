from __future__ import annotations
import os
import typer
from rich import print

from photofinder.pipeline import index_imagefolder
from photofinder.indexing.store import load_index_npz
from photofinder.eval.retrieval import eval_retrieval
from photofinder.utils import ensure_dir, write_json

app = typer.Typer(no_args_is_help=True)

@app.command()
def index(
    dataset: str = typer.Option(..., help="Path to dataset root: root/<label>/<image>"),
    model: str = typer.Option("dummy", help="Model name (dummy, dlib_resnet_v1, etc.)"),
    out: str = typer.Option(..., help="Output folder (will create index.npz)"),
):
    """Build an embedding index."""
    index_path = index_imagefolder(dataset_root=dataset, model_name=model, out_dir=out)
    print(f"[green]Saved:[/green] {index_path}")

@app.command("eval-retrieval")
def eval_retrieval_cmd(
    index: str = typer.Option(..., help="Path to index.npz"),
    out: str = typer.Option(..., help="Output folder for metrics.json"),
):
    """Evaluate retrieval (Rank-1/Recall@k/MRR) on identities with >=2 images."""
    ensure_dir(out)
    emb, meta = load_index_npz(index)
    m = eval_retrieval(emb, meta, top_k=10)
    metrics = m.__dict__
    write_json(os.path.join(out, "metrics_retrieval.json"), metrics)
    print(metrics)

if __name__ == "__main__":
    app()

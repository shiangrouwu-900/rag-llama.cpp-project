import argparse
import csv
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_QUERY = "AORUS MASTER 16 AM6H 的重量和尺寸是多少?"
DEFAULT_VRAM_LIMIT_MB = 4096


@dataclass
class VramSample:
    timestamp: float
    used_mb: int
    total_mb: int
    stage: str


@dataclass
class VramMonitor:
    interval: float = 0.2
    gpu_index: int = 0
    samples: list[VramSample] = field(default_factory=list)
    _stage: str = "idle"
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def set_stage(self, stage: str):
        self._stage = stage
        self.sample_once()

    def sample_once(self):
        sample = query_gpu_memory(self.gpu_index, self._stage)
        if sample is not None:
            self.samples.append(sample)

    def _run(self):
        while not self._stop_event.is_set():
            self.sample_once()
            self._stop_event.wait(self.interval)

    def peak(self):
        if not self.samples:
            return None
        return max(self.samples, key=lambda sample: sample.used_mb)

    def peak_by_stage(self):
        peaks = {}
        for sample in self.samples:
            current = peaks.get(sample.stage)
            if current is None or sample.used_mb > current.used_mb:
                peaks[sample.stage] = sample
        return peaks


def query_gpu_memory(gpu_index: int, stage: str):
    command = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    first_line = output.strip().splitlines()[0]
    used_text, total_text = [part.strip() for part in first_line.split(",")]
    return VramSample(
        timestamp=time.time(),
        used_mb=int(used_text),
        total_mb=int(total_text),
        stage=stage,
    )


def save_csv(samples: list[VramSample], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "stage", "used_mb", "total_mb"])
        for sample in samples:
            writer.writerow([sample.timestamp, sample.stage, sample.used_mb, sample.total_mb])


def format_mb(value):
    return f"{value:,} MB"


def print_report(
    monitor: VramMonitor,
    baseline_mb: int | None,
    csv_path: Path | None,
    vram_limit_mb: int,
):
    if not monitor.samples:
        print("No VRAM samples were collected. Is NVIDIA driver / nvidia-smi available?")
        return

    print("\n=== VRAM report ===")
    if baseline_mb is not None:
        print(f"Baseline: {format_mb(baseline_mb)}")

    for stage, peak in monitor.peak_by_stage().items():
        delta = ""
        if baseline_mb is not None:
            delta = f" (+{format_mb(max(0, peak.used_mb - baseline_mb))})"
        print(f"{stage:20s} peak: {format_mb(peak.used_mb)} / {format_mb(peak.total_mb)}{delta}")

    overall = monitor.peak()
    if overall is not None:
        print(f"Overall peak:        {format_mb(overall.used_mb)} / {format_mb(overall.total_mb)} at stage '{overall.stage}'")
        measured_mb = overall.used_mb
        measured_label = "overall peak"
        if baseline_mb is not None:
            measured_mb = max(0, overall.used_mb - baseline_mb)
            measured_label = "peak increase from baseline"

        status = "PASS" if measured_mb <= vram_limit_mb else "FAIL"
        print(
            f"{vram_limit_mb} MB check:       {status} "
            f"({measured_label}: {format_mb(measured_mb)}, limit: {format_mb(vram_limit_mb)})"
        )

    if csv_path is not None:
        print(f"CSV saved: {csv_path}")


def add_project_to_path(project_dir: Path):
    project_dir = project_dir.resolve()
    os.chdir(project_dir)
    sys.path.insert(0, str(project_dir))
    return project_dir


def resolve_model_path(model_path: str | Path, project_dir: Path):
    path = Path(model_path)
    if not path.is_absolute():
        path = project_dir / path
    return path.resolve()


def run_rag(args, monitor: VramMonitor):
    project_dir = add_project_to_path(args.project_dir)

    from rag.embedding import load_embedding_model
    from rag.generation import build_prompt, generate_stream, load_llm
    from rag.retrieval import load_index, retrieve

    model_path = resolve_model_path(args.model_path, project_dir)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Use --model-path or set MODEL_PATH to the GGUF file path."
        )

    monitor.set_stage("load_index")
    embeddings, chunks = load_index(args.storage_dir)

    monitor.set_stage("load_embedding")
    embedding_model = load_embedding_model(args.embedding_model)

    monitor.set_stage("load_llm")
    llm = load_llm(
        model_path=str(model_path),
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
    )

    monitor.set_stage("retrieve")
    results = retrieve(
        query=args.query,
        model=embedding_model,
        embeddings=embeddings,
        chunks=chunks,
        top_k=args.top_k,
    )

    prompt = build_prompt(args.query, results)

    monitor.set_stage("generate")
    print("\n=== Model output ===")
    generate_stream(llm, prompt, args.max_tokens)

    monitor.set_stage("finished")


def parse_args():
    parser = argparse.ArgumentParser(description="Measure real GPU VRAM usage while running the RAG pipeline.")
    parser.add_argument("--project-dir", type=Path, default=Path.cwd(), help="Project root that contains main.py and rag/.")
    parser.add_argument("--storage-dir", default="storage", help="Index storage directory, relative to project-dir.")
    parser.add_argument("--model-path", default=os.getenv("MODEL_PATH", "models/Qwen3-1.7B-Q4_K_M.gguf"))
    parser.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", "4")))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", "192")))
    parser.add_argument("--n-ctx", type=int, default=int(os.getenv("N_CTX", "3072")))
    parser.add_argument("--n-gpu-layers", type=int, default=int(os.getenv("N_GPU_LAYERS", "-1")))
    parser.add_argument("--n-threads", type=int, default=int(os.getenv("N_THREADS", "4")))
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--csv", type=Path, default=Path("vram_samples.csv"), help="Where to save VRAM samples.")
    parser.add_argument("--vram-limit-mb", type=int, default=DEFAULT_VRAM_LIMIT_MB)
    return parser.parse_args()


def main():
    args = parse_args()
    monitor = VramMonitor(interval=args.sample_interval, gpu_index=args.gpu_index)

    baseline_sample = query_gpu_memory(args.gpu_index, "baseline")
    baseline_mb = baseline_sample.used_mb if baseline_sample is not None else None
    if baseline_sample is not None:
        monitor.samples.append(baseline_sample)

    monitor.start()
    try:
        run_rag(args, monitor)
    finally:
        monitor.stop()
        if args.csv:
            save_csv(monitor.samples, args.csv)
        print_report(monitor, baseline_mb, args.csv, args.vram_limit_mb)


if __name__ == "__main__":
    main()

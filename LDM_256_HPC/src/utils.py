"""
utils.py — Pomoćne funkcije za DDPM trening na Supek HPC (PBS Pro).

Pokriva:
  - Reproducibilnost  : seed_everything
  - Datotečni sustav  : ensure_dir
  - Preciznost        : detect_mixed_precision
  - GPU dijagnostika  : check_gpu, get_gpu_memory_summary
  - Inspekcija modela : count_parameters, format_param_count
  - DataLoader        : get_num_workers (PBS-aware)
  - Logging klastera  : log_cluster_env (PBS-aware)
"""
from __future__ import annotations

import os
import platform
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """
    Postavi seed za sve izvore slučajnosti (Python, NumPy, PyTorch).

    Ako je deterministic=True, cuDNN se prisilno prebacuje u potpuno
    deterministički način rada — svaka dva identična pokretanja daju
    numerički iste rezultate. Korisno za finalne eksperimente u tezi.
    Cijena je ~10-15% sporiji trening pa ga ne koristiti pri razvoju.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic

    # Čini Python hash() deterministički između procesa i pokretanja.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def ensure_dir(path: str) -> str:
    """
    Stvori direktorij (i sve roditelje) ako ne postoji.

    Vraća putanju kako bi je pozivatelj mogao koristiti inline, npr.:
        out_path = os.path.join(ensure_dir(cfg.out_dir), "results.json")
    """
    os.makedirs(path, exist_ok=True)
    return path


def detect_mixed_precision() -> str:
    """
    Automatski odredi optimalnu mixed precision za dostupni GPU.

    Vraća jedan od: 'bf16', 'fp16', 'no'.

    Logika odluke temelji se na CUDA compute capability:
      - bf16 (>= 8.0): A100, H100 — bf16 ima isti 8-bitni eksponent kao fp32
        pa gradienti gotovo nikad ne overflow-aju. Preporučeno za Supek.
      - fp16 (>= 7.0): V100, RTX 2080 — zahtijeva loss scaling.
      - no  (< 7.0 ili CPU): puna fp32 preciznost.
    """
    if not torch.cuda.is_available():
        return "no"

    props = torch.cuda.get_device_properties(0)
    cap = props.major * 10 + props.minor  # A100 (8.0) → 80, V100 (7.0) → 70

    if cap >= 80:
        return "bf16"
    if cap >= 70:
        return "fp16"
    return "no"


def check_gpu(min_memory_gb: float = 8.0) -> None:
    """
    Provjeri dostupnost GPU-a i logiraj dijagnostičke informacije pri startu.

    Baca RuntimeError s uputstvima ako CUDA nije dostupna ili ako GPU nema
    dovoljno ukupne memorije za trening. Upozorava (bez pada) ako je slobodna
    memorija kritično niska — što može ukazivati da drugi posao dijeli čvor.

    Provjera slobodne memorije (ne samo ukupne) je bitna jer na klasteru
    GPU čvor može biti djelomično zauzet, ostavljajući manje nego što
    naziv modela sugerira.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. On Supek, check all of the following:\n"
            "  1. PBS script requests a GPU:  #PBS -l select=1:ngpus=1\n"
            "  2. You submitted to GPU queue: #PBS -q gpu\n"
            "  3. CUDA module is loaded:      module load cuda  (check: module avail cuda)\n"
            "  4. Verify with nvidia-smi on the login node before submitting."
        )

    n_gpus = torch.cuda.device_count()
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set by PBS")
    print(f"[gpu] {n_gpus} GPU(s) visible | CUDA_VISIBLE_DEVICES={cuda_visible}")

    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1024**3

        torch.cuda.set_device(i)
        free_bytes, _ = torch.cuda.mem_get_info(i)
        free_gb = free_bytes / 1024**3
        used_gb = total_gb - free_gb

        print(
            f"[gpu:{i}] {props.name} | "
            f"Total: {total_gb:.1f} GB | "
            f"Used: {used_gb:.1f} GB | "
            f"Free: {free_gb:.1f} GB | "
            f"Compute: {props.major}.{props.minor} | "
            f"Best precision: {detect_mixed_precision()}"
        )

        if total_gb < min_memory_gb:
            raise RuntimeError(
                f"GPU {i} ({props.name}) has only {total_gb:.1f} GB total memory; "
                f"at least {min_memory_gb:.1f} GB required."
            )

        if free_gb < min_memory_gb * 0.6:
            print(
                f"[gpu:{i}] WARNING: only {free_gb:.1f} GB free before training starts. "
                "Another process may be using this GPU. "
                "Risk of OOM — consider qdel the job and resubmitting, "
                "or reduce batch_size / enable gradient checkpointing."
            )


def get_gpu_memory_summary(device: Optional[torch.device] = None) -> str:
    """
    Vrati jednolinski sažetak GPU memorije za periodično logiranje.

    Koristan za pozivanje na početku treninga i svakih N koraka kako bi se
    otkrili memory leakovi — stalni rast 'reserved' memorije je upozoravajući znak.

    Primjer ispisa: 'allocated=3.21 GB | reserved=4.00 GB | free=35.79 GB'
    """
    if not torch.cuda.is_available():
        return "CUDA not available"

    dev = device or torch.device("cuda")
    allocated_gb = torch.cuda.memory_allocated(dev) / 1024**3
    reserved_gb  = torch.cuda.memory_reserved(dev)  / 1024**3
    free_bytes, _ = torch.cuda.mem_get_info(dev)
    free_gb = free_bytes / 1024**3

    return (
        f"allocated={allocated_gb:.2f} GB | "
        f"reserved={reserved_gb:.2f} GB | "
        f"free={free_gb:.2f} GB"
    )


def count_parameters(model: nn.Module) -> dict[str, int]:
    """
    Prebroji trainable i zamrznute parametre modela.

    Vraća dictionary s ključevima 'trainable', 'frozen', 'total'.
    Broj parametara je standardni podatak koji mora biti prijavljen u tezi —
    omogućuje čitateljima da razumiju skalu modela i usporede arhitekture.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}


def format_param_count(n: int) -> str:
    """
    Formatiraj broj parametara kao čitljiv string.

    Primjeri: 35_400_000 → '35.4M', 512_000 → '512.0K', 128 → '128'.
    """
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def get_num_workers(cfg_value: int) -> int:
    """
    Odredi optimalni broj DataLoader workera za trenutno okruženje.

    Prioritetni redoslijed (prvo što odgovara pobjeđuje):

    1. PBS klaster (Supek): čita $NCPUS koji PBS postavlja prema broju
       CPU-ova zatraženih u #PBS -l select=1:ncpus=N. Rezervira 1 CPU
       za glavni proces, ostalo daje workerima.

    2. Windows: uvijek vraća 0 jer Windows koristi 'spawn' umjesto 'fork'
       za multiprocessing, što s PyTorch DataLoaderom često deadlocka.

    3. Lokalni GPU (razvoj): koristi cfg_value iz YAML konfiga.

    4. CPU-only fallback: vraća 0 — workeri ne pomažu bez GPU compute-a.
    """
    pbs_ncpus = os.environ.get("NCPUS")
    if pbs_ncpus is not None:
        try:
            allocated = int(pbs_ncpus)
            workers = max(1, allocated - 1)
            print(f"[workers] PBS: NCPUS={allocated} → using {workers} DataLoader workers")
            return workers
        except ValueError:
            print(f"[workers] Warning: $NCPUS='{pbs_ncpus}' is not a valid integer. Falling back.")

    if platform.system() == "Windows":
        return 0

    if torch.cuda.is_available():
        return cfg_value

    return 0


def log_cluster_env() -> None:
    """
    Logiraj ključne PBS varijable okoline i verzije softvera pri startu joba.

    Rezultat završava u PBS .o log datoteci i odgovara na prva pitanja pri
    debugiranju neuspješnog joba: koji čvor je radio, koji GPU je dodijeljen,
    koje verzije softvera su bile aktivne?

    Napomena: ova funkcija zamjenjuje stari log_slurm_env() — Supek koristi
    PBS Pro, ne SLURM. SLURM varijable nikad nisu postavljene na Supeku.
    """
    pbs_vars: dict[str, str] = {
        "PBS_JOBID":            "Job ID",
        "PBS_JOBNAME":          "Job name",
        "PBS_NODEFILE":         "Node file path",
        "NCPUS":                "CPUs allocated",
        "CUDA_VISIBLE_DEVICES": "GPU device(s)",
        "TMPDIR":               "Scratch directory",
    }

    found = {label: os.environ[key] for key, label in pbs_vars.items() if key in os.environ}

    if found:
        print("\n[cluster] === PBS Environment ===")
        for label, val in found.items():
            print(f"  {label}: {val}")
    else:
        print("[cluster] No PBS environment detected — running locally (no PBS vars set).")

    cuda_ver = torch.version.cuda or "N/A"
    print(
        f"[env] Python {platform.python_version()} | "
        f"PyTorch {torch.__version__} | "
        f"CUDA {cuda_ver}"
    )
    if torch.cuda.is_available():
        print(
            f"[env] cuDNN {torch.backends.cudnn.version()} | "
            f"Device: {torch.cuda.get_device_name(0)}"
        )
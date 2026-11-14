"""
Laboratorio 1 - LLMOps: Benchmarking de serving local y cuantización
=====================================================================
Modelo evaluado: Qwen3 0.6B (el modelo instructivo moderno más liviano que funciona en CPU)
Variantes:       q4_K_M (4 bits) vs q8_0 (8 bits)
Runtime:         Ollama en Docker, CPU-only
Contexto:        num_ctx = 2048 tokens

El script:
  1. Hace warm-up para cargar los pesos en memoria.
  2. Ejecuta 3 prompts (corto, medio, largo) por cada cuantización.
  3. Mide TTFT, TPS, tokens de entrada/salida y tiempo total.
  4. Muestrea la RAM del contenedor con `docker stats` durante cada prueba.
  5. Exporta resultados/resultados_benchmark.csv, resultados/resultados_benchmark.md
     y el resumen Metricas.md.
"""

import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from tabulate import tabulate

OLLAMA_URL = "http://localhost:11434/api/generate"
CONTAINER_NAME = "ollama-local-lab"
OUT_DIR = Path("resultados")
METRICAS_FILE = Path("Metricas.md")

MODELS = [
    {"tag": "qwen3:0.6b-q4_K_M", "label": "q4_K_M (4 bits)"},
    {"tag": "qwen3:0.6b-q8_0", "label": "q8_0 (8 bits)"},
]

PROMPTS = [
    {
        "tipo": "Corto (Definición)",
        "texto": "Explica en exactamente 3 viñetas qué es la cuantización en modelos de lenguaje.",
    },
    {
        "tipo": "Medio (Razonamiento)",
        "texto": (
            "Tengo un servidor con 16 GB de RAM y sin GPU. Quiero desplegar un asistente "
            "interno para 5 empleados. ¿Qué arquitectura y tamaño de modelo me recomiendas y por qué?"
        ),
    },
    {
        "tipo": "Largo (Generación de Código)",
        "texto": (
            "Escribe una función en Python bien documentada que implemente búsqueda binaria "
            "sobre una lista ordenada, incluyendo manejo de excepciones y 3 casos de prueba."
        ),
    },
]

# Desactiva la lectura de proxies del sistema: Ollama es local.
SESSION = requests.Session()
SESSION.trust_env = False


def container_mem_mb() -> float:
    """RAM usada por el contenedor según docker stats, en MB."""
    out = subprocess.check_output(
        [
            "docker", "stats", "--no-stream",
            "--format", "{{.MemUsage}}",
            CONTAINER_NAME,
        ],
        text=True,
        timeout=30,
    ).strip()
    used = out.split("/")[0].strip()
    value = float("".join(ch for ch in used if ch.isdigit() or ch == "."))
    if "GiB" in used or "GB" in used:
        return value * 1024
    if "KiB" in used or "KB" in used:
        return value / 1024
    return value  # MiB / MB


def stable_mem_mb(samples: int = 3) -> float:
    """Mediana de varias lecturas para reducir el ruido de docker stats."""
    values = []
    for _ in range(samples):
        try:
            values.append(container_mem_mb())
        except Exception:
            pass
        time.sleep(0.3)
    if not values:
        return 0.0
    values.sort()
    return values[len(values) // 2]


def parse_ollama_ps_size_mb(ps_output: str) -> float:
    """Extrae la columna SIZE de `ollama ps` (p. ej. '783 MB' o '1.1 GB')."""
    for line in ps_output.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].startswith("qwen"):
            for i, part in enumerate(parts):
                if part in ("MB", "GB") and i > 0:
                    try:
                        value = float(parts[i - 1])
                    except ValueError:
                        continue
                    return round(value * 1024 if part == "GB" else value, 1)
    return 0.0


class MemorySampler(threading.Thread):
    """Muestrea la memoria del contenedor mientras corre una inferencia."""

    def __init__(self, interval=0.4):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples = []

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.samples.append(container_mem_mb())
            except Exception:
                pass
            self.stop_event.wait(self.interval)

    def stop(self):
        self.stop_event.set()
        self.join(timeout=10)
        return max(self.samples) if self.samples else 0.0


def stop_all_models():
    """Descarga los modelos de RAM para medir una línea base limpia."""
    for model in MODELS:
        subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "ollama", "stop", model["tag"]],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    time.sleep(2)


def run_inference(model: str, prompt: str, num_predict: int = 512):
    """Ejecuta una inferencia en streaming y devuelve métricas + respuesta."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "think": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 2048,
            "num_predict": num_predict,
            "repeat_penalty": 1.1,
            "seed": 42,
        },
    }

    start_time = time.perf_counter()
    first_token_time = None
    full_response = []
    final_chunk = {}

    with SESSION.post(OLLAMA_URL, json=payload, stream=True, timeout=600) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            if first_token_time is None and chunk.get("response"):
                first_token_time = time.perf_counter()
            full_response.append(chunk.get("response", ""))
            if chunk.get("done"):
                final_chunk = chunk

    end_time = time.perf_counter()
    total_time_s = end_time - start_time
    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0.0

    eval_count = final_chunk.get("eval_count", 0)
    eval_duration_ns = final_chunk.get("eval_duration", 0)
    tps = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns else 0.0

    return {
        "ttft_ms": round(ttft_ms, 2),
        "tps": round(tps, 2),
        "tokens_generados": eval_count,
        "tokens_prompt": final_chunk.get("prompt_eval_count", 0),
        "tiempo_total_s": round(total_time_s, 2),
        "load_ms": round(final_chunk.get("load_duration", 0) / 1e6, 2),
        "prompt_eval_ms": round(final_chunk.get("prompt_eval_duration", 0) / 1e6, 2),
        "respuesta_completa": "".join(full_response).strip(),
        "done_reason": final_chunk.get("done_reason", ""),
    }


def ollama_ps() -> str:
    """Salida de `ollama ps` para evidenciar el modelo residente y el uso de CPU."""
    result = subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "ollama", "ps"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    return (result.stdout or result.stderr).strip()


def run_inference_with_retry(model: str, prompt: str, num_predict: int = 512, retries: int = 2):
    """Reintenta si el runner de Ollama aún está arrancando (error EOF transitorio)."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            return run_inference(model, prompt, num_predict=num_predict)
        except Exception as exc:  # requests + errores de conexión del runner
            last_error = exc
            if attempt == retries:
                raise
            print(f"\n  [reintento {attempt + 1}/{retries}] {type(exc).__name__}: {exc}")
            time.sleep(4)
    raise last_error


def main():
    resultados_tabular = []
    resultados_detallados = []
    memoria_por_modelo = []

    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 68)
    print(" BENCHMARK DE CUANTIZACION LOCAL - QWEN3 0.6B")
    print("=" * 68)

    for model in MODELS:
        tag, label = model["tag"], model["label"]
        print(f"\n--- Evaluando {tag} ({label}) ---")

        stop_all_models()
        baseline_mb = stable_mem_mb()
        print(f"RAM base del contenedor (sin modelo): {baseline_mb:.1f} MB")

        print("Warm-up: cargando pesos en memoria...")
        warmup = run_inference_with_retry(tag, "Hola", num_predict=8)
        time.sleep(1)
        resident_mb = stable_mem_mb()
        ps_output = ollama_ps()
        ram_ollama_mb = parse_ollama_ps_size_mb(ps_output)
        print(f"RAM con modelo residente: {resident_mb:.1f} MB")
        print(f"RAM reportada por ollama ps: {ram_ollama_mb:.1f} MB")
        print(f"Tiempo de carga (load_duration): {warmup['load_ms']:.0f} ms")

        for item in PROMPTS:
            print(f"  -> Prompt {item['tipo']}...", end=" ", flush=True)
            sampler = MemorySampler()
            sampler.start()
            metrics = run_inference_with_retry(tag, item["texto"])
            time.sleep(0.5)
            peak_mb = max(sampler.stop(), resident_mb)

            print(
                f"TTFT={metrics['ttft_ms']:.0f} ms | "
                f"TPS={metrics['tps']:.2f} | "
                f"tokens={metrics['tokens_generados']} | "
                f"pico RAM={peak_mb:.1f} MB"
            )

            resultados_tabular.append(
                {
                    "Modelo": tag.split(":")[1],
                    "Cuantizacion": label,
                    "Prompt": item["tipo"],
                    "TTFT (ms)": metrics["ttft_ms"],
                    "TPS": metrics["tps"],
                    "Tokens salida": metrics["tokens_generados"],
                    "Tokens prompt": metrics["tokens_prompt"],
                    "Tiempo total (s)": metrics["tiempo_total_s"],
                    "RAM residente (MB)": round(resident_mb, 1),
                    "RAM modelo Ollama (MB)": round(ram_ollama_mb, 1),
                    "RAM pico (MB)": round(peak_mb, 1),
                }
            )

            resultados_detallados.append(
                {
                    "modelo": tag,
                    "cuantizacion": label,
                    "tipo_prompt": item["tipo"],
                    "texto_prompt": item["texto"],
                    "respuesta_completa": metrics["respuesta_completa"],
                    "ttft_ms": metrics["ttft_ms"],
                    "tps": metrics["tps"],
                    "tokens": metrics["tokens_generados"],
                    "tokens_prompt": metrics["tokens_prompt"],
                    "tiempo_total_s": metrics["tiempo_total_s"],
                    "load_ms": metrics["load_ms"],
                    "prompt_eval_ms": metrics["prompt_eval_ms"],
                    "done_reason": metrics["done_reason"],
                    "ram_residente_mb": round(resident_mb, 1),
                    "ram_ollama_mb": round(ram_ollama_mb, 1),
                    "ram_pico_mb": round(peak_mb, 1),
                }
            )

        memoria_por_modelo.append(
            {
                "modelo": tag,
                "cuantizacion": label,
                "ram_base_mb": round(baseline_mb, 1),
                "ram_residente_mb": round(resident_mb, 1),
                "ram_ollama_mb": round(ram_ollama_mb, 1),
                "incremento_mb": round(resident_mb - baseline_mb, 1),
                "load_ms": warmup["load_ms"],
                "ollama_ps": ps_output,
            }
        )

    df = pd.DataFrame(resultados_tabular)

    # CSV resumen con el mismo esquema de la guía (6 columnas)
    csv_resumen = df.rename(
        columns={
            "TPS": "Velocidad (TPS)",
            "Tokens salida": "Tokens",
            "Tiempo total (s)": "Total (s)",
        }
    )[["Modelo", "Prompt", "TTFT (ms)", "Velocidad (TPS)", "Tokens", "Total (s)"]]
    csv_resumen.to_csv(OUT_DIR / "resultados_benchmark.csv", index=False)

    resumen = (
        df.groupby(["Modelo", "Cuantizacion"], as_index=False)
        .agg(
            TTFT_medio_ms=("TTFT (ms)", "mean"),
            TPS_medio=("TPS", "mean"),
            Tokens_medios=("Tokens salida", "mean"),
            RAM_residente_MB=("RAM residente (MB)", "max"),
            RAM_modelo_Ollama_MB=("RAM modelo Ollama (MB)", "max"),
            RAM_pico_MB=("RAM pico (MB)", "max"),
        )
        .round(2)
    )

    md = []
    md.append("# Reporte de Benchmark y Respuestas: LLMOps Lab 1\n")
    md.append(
        "Benchmark de cuantización ejecutado sobre **Qwen3 0.6B** "
        "(el modelo instructivo moderno más liviano y rápido que funciona en CPU), servido con "
        "Ollama en Docker y ejecutado **100% en CPU**.\n"
    )
    md.append(f"- **Fecha de ejecución:** {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    md.append(f"- **Modelos:** `{MODELS[0]['tag']}` vs `{MODELS[1]['tag']}`")
    md.append("- **Contexto:** `num_ctx = 2048`, `temperature = 0.1`, `seed = 42`")
    md.append(f"- **Contenedor:** `{CONTAINER_NAME}` (Ollama latest)\n")
    md.append("## Resumen de métricas\n")
    md.append("| Modelo | Cuantización | TTFT medio (ms) | TPS medio | Tokens medios | RAM residente (MB) | RAM modelo Ollama (MB) | RAM pico (MB) |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, row in resumen.iterrows():
        md.append(
            f"| {row['Modelo']} | {row['Cuantizacion']} | {row['TTFT_medio_ms']:.2f} | "
            f"{row['TPS_medio']:.2f} | {row['Tokens_medios']:.1f} | "
            f"{row['RAM_residente_MB']:.1f} | {row['RAM_modelo_Ollama_MB']:.1f} | "
            f"{row['RAM_pico_MB']:.1f} |"
        )
    md.append("\n## Memoria por modelo\n")
    md.append("| Modelo | RAM base (MB) | RAM residente (MB) | RAM `ollama ps` (MB) | Incremento (MB) | Carga (ms) |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for mem in memoria_por_modelo:
        md.append(
            f"| {mem['modelo']} | {mem['ram_base_mb']:.1f} | {mem['ram_residente_mb']:.1f} | "
            f"{mem['ram_ollama_mb']:.1f} | {mem['incremento_mb']:.1f} | {mem['load_ms']:.0f} |"
        )

    md.append("\n## Respuestas completas\n")
    for idx, r in enumerate(resultados_detallados, 1):
        md.append(f"### Prueba {idx}: {r['modelo']} — {r['tipo_prompt']}\n")
        md.append(
            f"- **Métricas:** `TTFT: {r['ttft_ms']} ms` | `TPS: {r['tps']}` | "
            f"`Tokens salida: {r['tokens']}` | `Tokens prompt: {r['tokens_prompt']}` | "
            f"`Total: {r['tiempo_total_s']} s` | `RAM pico: {r['ram_pico_mb']} MB`"
        )
        md.append(f"\n**Prompt enviado:**\n\n> {r['texto_prompt']}\n")
        md.append(f"**Respuesta generada:**\n\n```text\n{r['respuesta_completa']}\n```\n")
        md.append("---\n")

    with open(OUT_DIR / "resultados_benchmark.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # Resumen ejecutivo en Metricas.md
    metricas = [
        "## Resumen de métricas\n",
        "| Modelo | Prompt | TTFT (ms) | TPS | Tokens | Total (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in csv_resumen.iterrows():
        modelo = row["Modelo"].split("-")[-1]
        metricas.append(
            f"| {modelo} | {row['Prompt']} | {row['TTFT (ms)']:.2f} | "
            f"{row['Velocidad (TPS)']:.2f} | {row['Tokens']} | {row['Total (s)']:.2f} |"
        )
    with open(METRICAS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(metricas) + "\n")

    print("\n=== BENCHMARK FINALIZADO ===")
    print("Resumen de métricas:\n")
    print(tabulate(resumen, headers="keys", tablefmt="github", showindex=False, floatfmt=".2f"))
    print("\nArchivos generados:")
    print(f"  1. {OUT_DIR / 'resultados_benchmark.csv'}  (resumen tabular)")
    print(f"  2. {OUT_DIR / 'resultados_benchmark.md'}   (6 respuestas completas + métricas)")
    print(f"  3. {METRICAS_FILE}                        (tabla resumen para el informe)")


if __name__ == "__main__":
    main()

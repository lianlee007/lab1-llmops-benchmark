# Entregable Laboratorio 1: Benchmarking de Serving Local y Cuantización

- **Curso:** LLMOps, Serving Local y Arquitecturas RAG
- **Estudiante:** Julián Leonardo Cerón
- **Hardware Utilizado:** Intel Core i5-13420H (8 núcleos / 12 hilos), 15.7 GB RAM, Windows 11 Pro, sin GPU dedicada
- **Fecha:** 13 de noviembre de 2026
- **Resultados:** `Metricas.md`, `resultados/resultados_benchmark.csv` y `resultados/resultados_benchmark.md`

> **Nota metodológica:** se usó `qwen3:0.6b` (Q4_K_M vs Q8_0) por ser el modelo instructivo más liviano que permite comparar ambas cuantizaciones en este entorno. Se probó el modelo exacto de la guía (Qwen 2.5 1.5B): su Q4 funciona a ~25 TPS, pero el Q8 (1.8 GB) entra en swap en el VM de 6.8 GB, cae a ~2.8 TPS y una generación de 64 tokens supera los 600 s. Qwen 2.5 0.5B además falla al leer los metadatos GGUF (`cannot allocate memory`). Qwen3 0.6B permite completar Q4 y Q8 en las mismas condiciones (~50 y ~37 TPS, sin swap). La comparación sigue siendo válida porque ambas variantes son del mismo modelo.

---

## Análisis Técnico de Resultados

### 1. Impacto en Memoria (MEM USAGE y KV Cache)
> ¿Cuál fue la diferencia exacta en MB/GB de RAM consumida entre `q4_K_M` y `q8_0` al ejecutar `docker stats`? ¿Por qué el consumo en ejecución supera el tamaño del archivo en disco?

**Respuesta:**
La diferencia medida con `docker stats` fue de **135.9 MB** (989.6 MB en Q8_0 frente a 853.7 MB en Q4_K_M), es decir, Q8 consumió **15.9% más RAM residente**. Respecto de la línea base del contenedor (265.7 MB para Q4 y 273.7 MB para Q8), Q4 añadió **588.0 MB** y Q8 añadió **715.9 MB**. `ollama ps` reporta una diferencia mayor: **783.0 MB vs 1126.4 MB** (**343.4 MB más, 43.9%**), porque incluye pesos + KV cache + overhead del runner.

El consumo en ejecución supera el peso en disco (522 MB en Q4 y 832 MB en Q8) por varios componentes:
- **KV cache** para `num_ctx = 2048`: para Qwen3 0.6B (28 capas, 8 cabezas KV, dimensión 128, FP16) ocupa `2 × 28 × 8 × 128 × 2048 × 2 bytes ≈ 224 MiB`.
- **Buffers de cómputo** de llama.cpp (activaciones, logits, atención).
- **Tokenizer, grafo de inferencia y overhead del proceso runner**.
- **Page cache** que `docker stats` contabiliza al leer los pesos.

Con eso se explican los valores: Q4 = 522 MB (disco) + 224 MiB (KV) + ~37 MB de overhead ≈ **783 MB**; Q8 = 832 MB + 224 MiB + ~70 MB ≈ **1126 MB**.

---

### 2. Ancho de Banda de Memoria y Velocidad (TPS)
> En inferencia sobre CPU, el cuello de botella suele ser el ancho de banda de la memoria RAM. ¿En qué porcentaje aumentó la velocidad (TPS) al pasar de 8 bits a 4 bits?

**Respuesta:**
El TPS medio pasó de **37.61 tok/s** en Q8_0 a **50.35 tok/s** en Q4_K_M: un aumento de **+33.9%**. Por prompt: **+38.1%** en el corto, **+32.1%** en el medio y **+31.3%** en el largo.

La generación autoregresiva está limitada por el ancho de banda de memoria: en cada token hay que leer los pesos. Q4_K_M almacena aproximadamente la mitad de bits por peso que Q8_0, por lo que reduce el tráfico de memoria y acelera la generación. No se observa un 2× teórico porque el KV cache, los buffers y el cómputo no se reducen en la misma proporción, y el modelo de 0.6B dedica una fracción importante del tiempo a overhead fijo.

---

### 3. Latencia Inicial (TTFT) y Fase Prefill
> ¿Por qué el `TTFT (ms)` es significativamente más alto en el Prompt Largo (Generación de Código) que en el Prompt Corto (Definición)?

**Respuesta:**
En mis mediciones **no fue significativamente más alto**: Q4 midió 313 ms en el prompt corto y 338 ms en el largo (solo +25 ms); Q8 midió 423 ms en el largo y presentó un outlier de 1726 ms en el corto por arranque en frío. Los prompts son muy cortos (39, 66 y 52 tokens), así que el *prefill* es casi instantáneo y el TTFT está dominado por el overhead fijo (scheduling, tokenización y primer paso del grafo), no por la longitud.

El efecto teórico sí existe: el *prefill* procesa todos los tokens de entrada y su coste crece con el número de tokens (la atención, aproximadamente O(n²)). Con prompts de cientos o miles de tokens, el TTFT del prompt largo sería claramente mayor. El outlier de 1726 ms en Q8/corto corresponde a un arranque en frío del runner, no a la longitud del prompt.

---

### 4. Calidad vs. Eficiencia en `resultados/resultados_benchmark.md`
> Revisando las 6 respuestas completas generadas en `resultados/resultados_benchmark.md`, ¿hubo alguna degradación perceptible en la lógica del código Python o en el razonamiento arquitectónico generado por el modelo de 4 bits frente al de 8 bits?

**Respuesta:**
No se observa una degradación **sistemática** atribuible a la cuantización; las diferencias están dominadas por la capacidad del modelo de 0.6B. En **código**, ambas variantes implementan correctamente la búsqueda binaria. Q8 completó la función y los 3 casos de prueba dentro del límite de 512 tokens; Q4 se cortó a mitad de frase por el límite, aunque la lógica era correcta. Q4 maneja la lista vacía con `ValueError`; Q8 devuelve `-1` para lista vacía, lo que contradice su propio docstring (dice que lanza `ValueError` solo si no es una lista). Ninguna de las dos es perfecta. En **razonamiento arquitectónico**, ambas fallan e inventan modelos y tamaños (“7B + 2B”, “H1/H3”, “A100/H100”); es un límite de capacidad del modelo, no un efecto de Q4 vs Q8.

Por tanto, **Q4_K_M es la mejor relación calidad/eficiencia para CPU**: entrega la misma calidad práctica que Q8 en este modelo, con ~34% más TPS y ~16% menos RAM residente. Q8_0 solo se justificaría si una evaluación de dominio específica mostrara una mejora de calidad que compensara su mayor coste. Para un asistente real de 5 empleados, un modelo de 0.6B es insuficiente; se recomendaría un 1.5B–3B en Q4_K_M sobre CPU, o un 7B–8B en GPU.

---

*Nota: adjuntar junto con este documento los archivos generados `Metricas.md`, `resultados/resultados_benchmark.csv` y `resultados/resultados_benchmark.md`.*

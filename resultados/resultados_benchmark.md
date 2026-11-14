# Reporte de Benchmark y Respuestas: LLMOps Lab 1

Benchmark de cuantización ejecutado sobre **Qwen3 0.6B** (el modelo instructivo moderno más liviano y rápido que funciona en CPU), servido con Ollama en Docker y ejecutado **100% en CPU**.

- **Fecha de ejecución:** 2026-11-13T00:23:36-05:00
- **Modelos:** `qwen3:0.6b-q4_K_M` vs `qwen3:0.6b-q8_0`
- **Contexto:** `num_ctx = 2048`, `temperature = 0.1`, `seed = 42`
- **Contenedor:** `ollama-local-lab` (Ollama latest)

## Resumen de métricas

| Modelo | Cuantización | TTFT medio (ms) | TPS medio | Tokens medios | RAM residente (MB) | RAM modelo Ollama (MB) | RAM pico (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.6b-q4_K_M | q4_K_M (4 bits) | 339.56 | 50.35 | 396.0 | 853.7 | 783.0 | 943.8 |
| 0.6b-q8_0 | q8_0 (8 bits) | 887.20 | 37.61 | 419.7 | 989.6 | 1126.4 | 1025.0 |

## Memoria por modelo

| Modelo | RAM base (MB) | RAM residente (MB) | RAM `ollama ps` (MB) | Incremento (MB) | Carga (ms) |
|---|---:|---:|---:|---:|---:|
| qwen3:0.6b-q4_K_M | 265.7 | 853.7 | 783.0 | 588.0 | 8400 |
| qwen3:0.6b-q8_0 | 273.7 | 989.6 | 1126.4 | 715.9 | 2894 |

## Respuestas completas

### Prueba 1: qwen3:0.6b-q4_K_M — Corto (Definición)

- **Métricas:** `TTFT: 313.46 ms` | `TPS: 53.44` | `Tokens salida: 250` | `Tokens prompt: 39` | `Total: 5.24 s` | `RAM pico: 943.8 MB`

**Prompt enviado:**

> Explica en exactamente 3 viñetas qué es la cuantización en modelos de lenguaje.

**Respuesta generada:**

```text
**Viñeta 1:**  
La cuantización en modelos de lenguaje se refiere a la transformación de las representaciones de datos (como texto o imágenes) en una forma binaria, lo que permite reducir el número de posibles estados y mejorar la eficiencia del modelo. Esto es fundamental para optimizar procesos como la traducción, la generación de textos o la detección de patrones.

**Viñeta 2:**  
La cuantización en modelos de lenguaje implica que los datos se representan con valores discretos (como 0 y 1), lo que facilita la comprensión del modelo. Por ejemplo, en el contexto de la red neuronal, esta transformación permite que las capas neuronales procesen información más eficientemente.

**Viñeta 3:**  
La cuantización en modelos de lenguaje también implica que los datos se reducen a una cantidad limitada, lo que mejora la precisión y la velocidad del modelo. Esto es clave para aplicaciones como la traducción natural o la generación de textos, donde el rendimiento puede ser significativamente mejorado sin perder información.
```

---

### Prueba 2: qwen3:0.6b-q4_K_M — Medio (Razonamiento)

- **Métricas:** `TTFT: 367.52 ms` | `TPS: 48.77` | `Tokens salida: 426` | `Tokens prompt: 66` | `Total: 9.19 s` | `RAM pico: 870.3 MB`

**Prompt enviado:**

> Tengo un servidor con 16 GB de RAM y sin GPU. Quiero desplegar un asistente interno para 5 empleados. ¿Qué arquitectura y tamaño de modelo me recomiendas y por qué?

**Respuesta generada:**

```text
Para tu servidor con **16 GB de RAM** y sin GPU, es recomendable implementar un **asistente interno de alto rendimiento** que combine:

- **Modelo adecuado para el entorno**
- **Tamaño de modelo optimizado**
- **Arquitectura eficiente**

### Recomendación:

#### 1. **Modelo:**
- **Llama: "LLama" (Hugging Face)**
- **Tamaño: 7B + 2B (total de 9B)**  
  - Es el modelo más popular y esférico, ideal para entornos como tu servidor.

#### 2. **Arquitectura:**
- **Modelo de GPU:** Si tu servidor tiene un GPU, usarlo para mejorar la rendimiento.
- **No usar GPU:**  
  - Para un entorno sin GPU, se recomienda usar **modeling without GPU** (no requiere hardware).  
  - Esto reduce costos y permite uso en sistemas con recursos limitados.

#### 3. **Tamaño de modelo:**
- **9B + 2B (total de 11B)**  
  - Es el tamaño más amplio para entornos como tu servidor, garantizando una respuesta rápida y alta calidad.
  - Aunque es un tamaño grande, no requiere GPU.

### Por qué?

- **Rendimiento:** El modelo tiene capacidad para procesar múltiples empleados simultáneamente (5 personas).
- **Costo:** Sin GPU, el tiempo de entrenamiento se reduce significativamente.
- **Sostenibilidad:** Aunque el tamaño es grande, no requiere hardware, lo que mejora la eficiencia del sistema.

### Conclusión:

Para un entorno con 16 GB de RAM y sin GPU, recomiendo usar un modelo de **9B + 2B (total de 11B)**. Esto garantiza una respuesta rápida, alta calidad y optimización para el rendimiento del servidor.
```

---

### Prueba 3: qwen3:0.6b-q4_K_M — Largo (Generación de Código)

- **Métricas:** `TTFT: 337.71 ms` | `TPS: 48.85` | `Tokens salida: 512` | `Tokens prompt: 52` | `Total: 10.89 s` | `RAM pico: 853.7 MB`

**Prompt enviado:**

> Escribe una función en Python bien documentada que implemente búsqueda binaria sobre una lista ordenada, incluyendo manejo de excepciones y 3 casos de prueba.

**Respuesta generada:**

```text
¡Claro! Aquí tienes una función bien documentada que implementa la búsqueda binaria sobre una lista ordenada en Python. Esta función maneja excepciones y incluye tres casos de prueba para verificar su funcionalidad.

```python
def binary_search(sorted_list, target):
    """
    Busca un elemento en una lista ordenada usando búsqueda binaria.
    
    Args:
        sorted_list: Lista ordenada (list).
        target: Valor que se busca.
        
    Returns:
        El índice del elemento encontrado o -1 si no se encuentra.
        
    Raises:
        ValueError: Si la lista está vacía.
    """
    # Verificar que la lista no esté vacía
    if not sorted_list:
        raise ValueError("La lista está vacía.")
    
    left = 0
    right = len(sorted_list) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        # Verificar si el valor medio es igual al objetivo
        if sorted_list[mid] == target:
            return mid
            
        # Si el valor buscado es menor que el medio, se busca a la derecha
        elif sorted_list[mid] < target:
            left = mid + 1
            
        else:  # Si el valor buscado es mayor que el medio, se busca a la izquierda
            right = mid - 1
            
    return -1
```

### Casos de Prueba:

```python
# Prueba con valor existente
print(binary_search([2, 4, 5, 7], 5))  # Devuelve: 2

# Prueba con valor no encontrado
print(binary_search([2, 4, 5, 7], 6))   # Devuelve: -1

# Prueba con lista vacía
try:
    print(binary_search([], 3))
except ValueError as e:
    print("Error:", e)
```

### Descripción:

- **Funcionalidad**: La función busca un elemento en una lista ordenada usando búsqueda binaria.
- **Manejo de excepciones**: Se detecta la lista vacía y lanzamos una excepción `ValueError`.
- **Criterios de prueba**:
  - Busca un valor existente.
  - Devuelve el índice del elemento encontrado o `-1` si no se encuentra.
  - Verifica que la lista esté vacía en caso de llamada direct
```

---

### Prueba 4: qwen3:0.6b-q8_0 — Corto (Definición)

- **Métricas:** `TTFT: 1726.46 ms` | `TPS: 38.7` | `Tokens salida: 260` | `Tokens prompt: 39` | `Total: 8.7 s` | `RAM pico: 1025.0 MB`

**Prompt enviado:**

> Explica en exactamente 3 viñetas qué es la cuantización en modelos de lenguaje.

**Respuesta generada:**

```text
**Viñeta 1:**  
En un modelo de lenguaje, la **cuantificación** implica que los datos se representan como **números binarios** (0 o 1) para simplificar procesos matemáticos y reducir recursos. Por ejemplo, en un modelo de red neuronal, las características del texto son codificadas como vectores numéricos, lo que permite optimizar la computación.

**Viñeta 2:**  
La cuantización **reduce el espacio de posibles estados** en un modelo de lenguaje. Esto mejora la precisión y la eficiencia al evitar valores no relevantes (como números aleatorios o fracciones). Por ejemplo, en un modelo de clasificación, los datos se codifican como entornos binarios para facilitar la interpretación.

**Viñeta 3:**  
La cuantización **permite que el modelo procese información con menos recursos**, lo cual es importante para modelos pequeños o con alto rendimiento. Por ejemplo, en un modelo de generación automática, la cuantificación reduce la complejidad del proceso de aprendizaje, permitiendo que se entrenen más rápido y con mayor precisión.
```

---

### Prueba 5: qwen3:0.6b-q8_0 — Medio (Razonamiento)

- **Métricas:** `TTFT: 512.48 ms` | `TPS: 36.93` | `Tokens salida: 512` | `Tokens prompt: 66` | `Total: 14.51 s` | `RAM pico: 989.6 MB`

**Prompt enviado:**

> Tengo un servidor con 16 GB de RAM y sin GPU. Quiero desplegar un asistente interno para 5 empleados. ¿Qué arquitectura y tamaño de modelo me recomiendas y por qué?

**Respuesta generada:**

```text
Para tu servidor con **16 GB de RAM** y **sin GPU**, es recomendable implementar un **asistente interno para 5 empleados** que combine una arquitectura eficiente, tamaño adecuado y funcional. Aquí te explico la opción más adecuada:

---

### ✅ **Arquitectura recomendada:**

- **Tamaño de modelo:**  
  - **Llama de alto rendimiento (H1):** Ideal para procesar lenguas multilingües, como Spanish, English, etc.  
  - **Tamaño de modelo con GPU:** No es compatible si no tienes un GPU.

- **Arquitectura:**  
  - **NVIDIA A100 o H100 (con GPU):** Si tu servidor tiene una GPU, pero si no, opta por un modelo con CPU y RAM.  
  - **Tamaño de modelo con CPU solo:**  
    - **Llama de alto rendimiento (H1):** Ideal para procesar lenguas multilingües.  
    - **Tamaño de modelo con CPU solo:**  
      - **Llama de alto rendimiento (H1):** Similar al anterior, pero con CPU solo.

---

### 📊 **Tamaño del modelo:**

- **H1 (2048 tokens):**  
  - Ideal para procesar lenguas multilingües y realizar tareas complejas.  
  - Tamaño de memoria: 2 GB (CPU + RAM).  
  - Es el tamaño más común en asistentes internos.

- **H3 (1024 tokens):**  
  - Menor tamaño, pero con mayor capacidad para lenguas multilingües y procesamiento.  
  - Tamaño de memoria: 1 GB (CPU + RAM).  
  - Ideal si necesitas más recursos en el modelo.

---

### 🧠 **Por qué elegir H1 o H3?**

- **H1:**  
  - Sencillo, rápido y adecuado para procesar lenguas multilingües.  
  - Tamaño de memoria: 2 GB (CPU + RAM).  
  - Ideal para un asistente interno con 5 empleados.

- **H3:**  
  - Menor tamaño, pero más potente en términos de capacidad de procesamiento.  
  - Tamaño de memoria:
```

---

### Prueba 6: qwen3:0.6b-q8_0 — Largo (Generación de Código)

- **Métricas:** `TTFT: 422.67 ms` | `TPS: 37.21` | `Tokens salida: 487` | `Tokens prompt: 52` | `Total: 13.63 s` | `RAM pico: 989.6 MB`

**Prompt enviado:**

> Escribe una función en Python bien documentada que implemente búsqueda binaria sobre una lista ordenada, incluyendo manejo de excepciones y 3 casos de prueba.

**Respuesta generada:**

```text
¡Claro! Aquí tienes una función bien documentada en Python que implementa la búsqueda binaria sobre una lista ordenada, incluyendo manejo de excepciones y tres casos de prueba.

```python
def binary_search(sorted_list, target):
    """
    Realiza una búsqueda binaria en una lista ordenada para encontrar el valor `target`.

    Args:
        sorted_list: Lista ordenada (de tipo list).
        target: Valor que se busca en la lista.

    Returns:
        El índice del elemento `target` si existe, o -1 si no lo encuentra.

    Raises:
        ValueError: Si `sorted_list` no es una lista.
    """
    
    # Verificar que el argumento sea un iterable
    if not isinstance(sorted_list, list):
        raise ValueError("La lista debe ser de tipo list.")
    
    # Verificar que la lista esté ordenada
    if sorted_list is None or len(sorted_list) == 0:
        return -1
    
    # Búsqueda binaria por valor
    left = 0
    right = len(sorted_list) - 1

    while left <= right:
        mid = (left + right) // 2

        if sorted_list[mid] == target:
            return mid
        elif sorted_list[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    # Si no se encuentra el valor, devolver -1
    return -1
```

### Casos de prueba:

```python
# Prueba 1: Valor existe en la lista
print(binary_search([1, 3, 5, 7, 9], 5))  # Devuelve 2

# Prueba 2: Valor no está presente
print(binary_search([1, 3, 5, 7, 9], 6))  # Devuelve -1

# Prueba 3: Lista vacía o sin elementos
try:
    print(binary_search([], 0))
except ValueError as e:
    print(f"Error: {e}")
```

### Notas importantes:

- La lista debe ser ordenada.
- Se manejan excepciones al pasar argumentos no válidos (como `None`, `str`).
- El código es bien documentado y fácil de entender.
```

---

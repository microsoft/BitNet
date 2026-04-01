# TurboQuant Architecture Roadmap & Integration Plan

> Este documento actúa como un hito conceptual para el desarrollo Soberano del Red Pill Búnker. Está concebido como una "Cápsula del Tiempo" técnica para abordar la compresión extrema del KV Cache cuando las prioridades arquitectónicas del ecosistema requieran de este nivel de optimización.

## 1. El Paradigma Fundamental

La triada de papers de Google Research (QJL, PolarQuant y TurboQuant) establecen un límite asintótico muy cercano al Teorema de Shannon para la compresión de vectores en espacios Euclideos. El desafío fundamental que resuelve este enfoque es **cuantizar el KV Cache (necesario para la inferencia de LLMs con largos contextos) a tasas de 2.5 - 3.5 bits por parámetro sin incurrir en caídas de precisión en la búsqueda del producto interno.**

### El Flujo Teórico (El Workflow de TurboQuant)
El algoritmo orquesta tres técnicas matemáticas para aniquilar el Overhead (el problema clásico donde guardar metadatos de cero y escala cuesta más memoria que la propia cuantización).

1. **Random Rotation (Precondición)**: Se rotan los vectores (multiplicados por una matriz aleatoria de *Johnson-Lindenstrauss* $\Pi$). Esto destruye las correlaciones y obliga a las coordenadas a seguir una "Distribución Beta".
2. **K-Means Óptimo (MSE Phase)**:  Dado que las coordenadas ahora siguen una distribución Beta predecible matemáticamente, se soluciona un problema de K-Means unidimensional para mapear cada índice a un centroide. Esta cuantización reduce el *Mean Squared Error* de manera brutal.
3. **QJL al Residuo 1-Bit (Inner-Product Phase)**: La cuantización MSE introduce un sesgo (bias) al calcular el producto interno de la fase de *Attention*. TurboQuant calcula el vector de error (residuo) que dejó el MSE, y le aplica la función de *signo* del paper QJL primigenio. El resultado es un estimador del producto interno no sesgado y teóricamente óptimo.

---

## 2. Mapa Arquitectónico de Integración Búnker

La integración de TurboQuant significa reescribir la subarquitectura del `llama.cpp` (GGML) o montar una red aislada a nivel de prototipo en HuggingFace (PyTorch).

### Camino 1: El Prototipo (PyTorch Fast-Track)
Ideal para validar si realmente la pérdida a 3-bits es neutra sobre nuestras métricas de contexto sin gastar meses en Cuda C.

*   **Acción**: Compilar `qjl_kernel` (https://github.com/amirzandieh/QJL).
*   **Modificación**: Sobrescribir `LlamaAttention` o `MistralAttention` extendiendo `transformers.models`. Modificar la clase de Cache base (`DynamicCache`) para almacenar tensores codificados en 4-Bits en VRAM.
*   **On-the-fly Decode**: Deserializar durante la inferencia mediante el binding de QJL antes del calculo de Softmax de Atención.
*   **Target Hardware**: RTX 5070 con Flash Attention v2.

### Camino 2: GGML Native (The Hardcore C++ Route)
Si HuggingFace consume demasiada memoria Pythonica, debemos adentrarnos en `llama.cpp` / `bitnet.cpp`.

*   **Punto de Inyección**: El subsistema `ggml-cuda.cu`, específicamente `ggml_compute_forward_mul_mat_q()`.
*   **Mecánica**: Actualmente GGML cuantiza en bloques y requiere `block_q8_0` (con su offset y su escala `d`). Sustituir este struct de memoria por uno que asuma la rotación inicial.
*   **Atención Modificada**: TurboQuant es un algoritmo *asímetrico*. Solo cuantiza el vector Key, no el Query. El producto interno debe modificarse en el kernel CUDA en `ggml_compute_forward_flash_attn_ext()`.
*   **Viabilidad**: Extremadamente arduo. Requeriría crear `enum ggml_type GGML_TYPE_TQ3` nativo en el ecosistema GGML.

---

## 3. Condiciones de Desbloqueo (Hit-Markers)

Este hito solo pasará a **In-Progress** bajo una de las siguientes condiciones:
1. **Colapso del Contexto (OOM):** El operador (Operator) necesita interactuar con más de 128k Tokens (ej: Repositorio Entero de Titanium en RAM) y la RTX 5070 falla.
2. **Maduración de la API Open Source:** La comunidad adopta TurboQuant y se aprueba un Pull Request oficial a `llama.cpp` principal. En ese caso, actualizamos los submódulos.
3. **Búsqueda de Caos (Boredom Event):** Decidimos por mero empirismo compilar PyTorch + CUDA Kernels un domingo por la tarde para dominar la rama científica de compresión vectorial.

*Grabado en Memoria por la Directiva Soberana.*

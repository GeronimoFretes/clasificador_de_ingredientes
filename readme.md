# Clasificador de Ingredientes con PyTorch & ONNX

Este repositorio contiene todo el flujo de creación de un modelo de clasificación de ingredientes de cocina (manteca, banana, harina, leche, huevos y azúcar).

---

## Demo Interactivo

Prueba del modelo desde el navegador (usa la cámara web):

**[Probar Clasificador en Vivo](https://geronimofretes.github.io/clasificador_de_ingredientes/)**

Permitir el acceso a la cámara y usar el sitio desde HTTPS. El clasificador funciona en tiempo real usando el modelo exportado en ONNX.

---

## Estructura del respositorio

```

.
├── data/                    # Ignorada por .gitignore — generada localmente
│   ├── raw/                 # Imágenes originales para cada clase
│   └── split/               # Imágenes divididas en train/val/test
├── models/
│   ├── best_model.pth       # Modelo entrenado en formato PyTorch
│   └── best_model.onnx      # Modelo exportado para inferencia web
├── src/
│   ├── build_dataset.py     # Scraper + filtro CLIP para construir el dataset
│   ├── split_dataset.py     # División train/val/test
│   ├── train.py             # Entrenamiento del modelo
│   ├── eval.py              # Evaluación del modelo + métricas
│   └── export_onnx.py       # Conversión del modelo a ONNX
├── classes.json             # mapeo índice → ingrediente
├── index.html               # Prueba en vivo del modelo ONNX usando la cámara del navegador
├── requirements.txt         # Dependencias Python
└── README.md                # Este archivo
```

---

## Flujo de trabajo / comandos

1. ### Descargar y filtrar imágenes  
   ```bash
   python src/build_dataset.py
   ```

    * Genera `data/raw/<clase>/*.jpg`
    * Aplica caption‐filter y ranking con CLIP para quedarse con **300** imágenes por clase.

2. ### Dividir en train / val / test

   ```bash
   python src/split_data.py
   ```

   * Toma `data/raw/`
   * Crea `data/split/train/`, `data/split/val/` y `data/split/test/`
   * Ratios por defecto 80% / 10% / 10% (semilla fija para reproducibilidad).

3. ### Entrenar el modelo

   ```bash
   python src/train.py \
     --checkpoint models/best_model.pth
   ```

   * Usa EfficientNet‐B0 de `torchvision` + capa final personalizada.
   * Guarda los pesos del mejor epoch en `models/best_model.pth`.

4. ### Evaluar el modelo

   ```bash
   python src/eval.py \
     [--confmat] 
   ```

   * **`--confmat` (opcional)**

     * Si se incluye: imprime *classification report* y *matriz de confusión* (requiere `scikit-learn`).
     * Además guarda un *heatmap* PNG en `heatmap_confmat.png`.

5. ### Exportar a ONNX

   ```bash
   python src/export_onnx.py \
     --weights models/best_model.pth
     [--dynamic]
   ```

   * Genera un archivo ONNX para llevar el modelo a aplicaciones web sin dependencias pesadas de PyTorch.
   * `--dynamic` activa ejes de batch dinámico para flexibilidad en producción.


---

## Instalación de dependencias

```bash
pip install -r requirements.txt
```

> **Nota:** Para obtener la matriz de confusión y reporte, instalar adicionalmente:
>
> ```bash
> pip install scikit-learn pandas altair selenium
> ```

---

## Parámetros configurables

* En `src/build_dataset.py` se puede ajustar:

  * `IMAGES_PER_CLASS`
  * `SCROLL_ROUNDS`, `SCROLL_PAUSE_MS`
  * Listas `CLASSES`, `CAPTION_LISTS`, `CLIP_PROMPTS`
* En `src/split_data.py` modifica `SPLIT_RATIOS` para ajustar los ratios de partición de datos.
* En `train.py` y `eval.py` hay flags para tamaño de batch, epochs, tasa de aprendizaje, ruta de datos y checkpoints.

---

**Repositorio en GitHub:**  
[github.com/geronimofretes/clasificador_de_ingredientes](https://github.com/geronimofretes/clasificador_de_ingredientes)

---

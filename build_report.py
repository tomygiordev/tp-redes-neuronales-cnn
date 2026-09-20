"""Build a five-page Spanish report exclusively from completed experiment outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("report/informe_cnn.pdf"))
    args = parser.parse_args()
    result = json.loads((args.results / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.results / "data_manifest.json").read_text(encoding="utf-8"))
    folds = result["folds"]
    if len(folds) != 5:
        raise ValueError("The report requires all five completed validation folds.")
    test = result["test"]
    config = result["config"]
    classes = manifest["class_names"]
    width = A4[0] - 88
    navy, teal = colors.HexColor("#17324D"), colors.HexColor("#087F8C")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("Body", fontName="Helvetica", fontSize=9.2, leading=12.3,
                              spaceAfter=7, textColor=colors.HexColor("#253342")))
    styles.add(ParagraphStyle("TitleCustom", fontName="Helvetica-Bold", fontSize=23,
                              leading=27, textColor=navy, spaceAfter=13))
    styles.add(ParagraphStyle("Section", fontName="Helvetica-Bold", fontSize=14,
                              leading=18, textColor=navy, spaceBefore=7, spaceAfter=10))
    styles.add(ParagraphStyle("SmallCustom", fontName="Helvetica", fontSize=8,
                              leading=10.2, spaceAfter=5, textColor=colors.HexColor("#425466")))
    styles.add(ParagraphStyle("CellCustom", fontName="Helvetica", fontSize=8.2, leading=10.2))
    story = []

    def paragraph(text, style="Body"):
        story.append(Paragraph(text, styles[style]))

    def heading(text):
        paragraph(text, "Section")

    def table(rows, widths=None):
        cells = [[Paragraph(escape(str(cell)), styles["CellCustom"]) for cell in row] for row in rows]
        item = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6EFF5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, 0), (-1, 0), .6, teal),
            ("LINEBELOW", (0, 1), (-1, -1), .25, colors.HexColor("#D8E1E8")),
        ]))
        story.extend([item, Spacer(1, 8)])

    def plot(filename, max_width, max_height):
        path = args.results / filename
        image_width, image_height = ImageReader(str(path)).getSize()
        scale = min(max_width / image_width, max_height / image_height)
        story.append(Image(str(path), width=image_width * scale, height=image_height * scale))

    def pct(value):
        return f"{100 * value:.2f}%"

    def metric_row(name, metrics):
        return [name, pct(metrics["accuracy"]), f"{metrics['macro_f1']:.4f}", f"{metrics['cross_entropy']:.4f}"]

    # Page 1: problem, data and safeguards.
    paragraph("TRABAJO PRÁCTICO INTEGRADOR · PARTE 1", "SmallCustom")
    paragraph("Clasificación de imágenes<br/>con una CNN", "TitleCustom")
    paragraph("Inteligencia artificial · Licenciatura en Sistemas", "SmallCustom")
    heading("1. Problema y alcance")
    paragraph("Se implementó y entrenó una red neuronal convolucional para clasificar imágenes RGB de "
              "32 × 32 píxeles en las diez categorías de CIFAR-10. La variable de salida es una clase "
              "mutuamente excluyente; la red produce probabilidades mediante softmax. El objetivo es "
              "evaluar generalización con validación por grupos, estudiar empíricamente sesgo y varianza "
              "y entregar un modelo Keras reutilizable. No se aborda la parte de redes recurrentes.")
    heading("2. Datos y prevención de filtraciones")
    duplicate = manifest["deduplication"]
    paragraph(f"CIFAR-10 contiene originalmente 50.000 imágenes de entrenamiento y 10.000 de prueba [1]. "
              f"Antes de muestrear, se detectaron imágenes con píxeles idénticos mediante SHA-256: "
              f"se descartaron {duplicate['train_removed']} de entrenamiento y {duplicate['test_removed']} "
              "de prueba. Se conserva la primera aparición y se da prioridad a entrenamiento, sin usar "
              "etiquetas para decidir qué eliminar. Esto evita duplicados exactos entre particiones, "
              "pero no detecta imágenes casi idénticas ni demuestra independencia semántica.")
    table([["Propiedad", "Ejecución documentada"],
           ["Desarrollo utilizado", f"{manifest['development_count']:,} imágenes; selección estratificada dentro de cada lote"],
           ["Prueba retenida", f"{manifest['test_count']:,} imágenes; consultada tras fijar arquitectura y épocas"],
           ["Grupos", "Los cinco archivos originales data_batch_1 a data_batch_5"],
           ["Clases", ", ".join(classes)],
           ["Semilla y presupuesto", f"Semilla base {config['seed']}; máximo {config['epochs']} épocas; minibatch {config['batch_size']}"],
           ["Escala de entrada", "RGB 0-255; Rescaling(1/255) incorporado al modelo"]], [120, width - 120])
    paragraph("GroupKFold deja un lote completo como validación y utiliza los otros cuatro para aprender [4]. "
              "Cada imagen de desarrollo se valida una vez. Los lotes son grupos de almacenamiento, "
              "no identificadores de sujeto, captura o procedencia: esta separación no garantiza "
              "independencia por fuente. La estratificación se aplica al muestreo dentro de cada lote; "
              "GroupKFold no asegura proporciones de clases idénticas entre folds.")
    test_scope = ("La eliminación de duplicados modifica la prueba: sus métricas no son directamente "
                  "comparables con resultados sobre las 10.000 imágenes oficiales completas. "
                  if duplicate["test_removed"] else
                  "No se eliminaron imágenes de prueba: se preservan las 10.000 imágenes oficiales. ")
    paragraph("El muestreo reducido permite reproducir el experimento con recursos moderados. " + test_scope +
              f"Se entrena con {manifest['development_count']:,} imágenes de desarrollo; comparar con "
              "trabajos que usan 50.000 exige considerar ese presupuesto diferente. "
              "El manifiesto guarda índices de selección y hashes de los píxeles para auditar la ejecución.")
    story.append(PageBreak())

    # Page 2: architecture and training mathematics.
    heading("3. Arquitectura y fundamentos")
    paragraph("Una capa convolucional comparte filtros en todas las posiciones, reduciendo parámetros "
              "y favoreciendo patrones locales. En Keras, Conv2D implementa correlación cruzada: "
              "y[i,j,f] = b[f] + sum(u,v,c) W[u,v,c,f] x[i·s+u-p,j·s+v-p,c]. "
              "La convolución matemática estricta invierte el filtro; en aprendizaje sus pesos libres "
              "hacen habitual llamar convolución a esta operación [3].")
    paragraph("Por dimensión espacial, n_out = floor((n + 2p - k)/s) + 1. Con k = 3, s = 1 y "
              "padding='same', p = 1 conserva ancho y alto. Cada MaxPooling2D usa ventana 2 × 2 y "
              "stride 2: reduce cada dimensión a la mitad, conserva el máximo local y no agrega pesos.")
    table([["Capa", "Salida", "Parámetros"],
           ["Entrada + Rescaling", "32 × 32 × 3", "0"],
           ["Conv2D 16, 3 × 3 + ReLU", "32 × 32 × 16", "448"],
           ["MaxPool 2 × 2", "16 × 16 × 16", "0"],
           ["Conv2D 32, 3 × 3 + ReLU", "16 × 16 × 32", "4.640"],
           ["MaxPool 2 × 2", "8 × 8 × 32", "0"],
           ["Conv2D 64, 3 × 3 + ReLU", "8 × 8 × 64", "18.496"],
           ["MaxPool 2 × 2 + Flatten", "4 × 4 × 64 → 1.024", "0"],
           ["Dense 64 + ReLU + Dropout(0,3)", "64", "65.600"],
           ["Dense 10 + Softmax", "10", "650"],
           ["Total entrenable", "", "89.834"]], [width * .49, width * .32, width * .19])
    paragraph("Cada convolución tiene (k²·C_entrada + 1)·C_salida parámetros; la densa tiene "
              "(N_entrada + 1)·N_salida. ReLU(z) = max(0,z) posee derivada 1 para z positivo y "
              "0 para z negativo: evita la saturación positiva de sigmoid, aunque puede producir "
              "neuronas inactivas. He normal usa varianza 2/fan_in y ayuda a conservar la escala de "
              "activaciones y gradientes. No garantiza por sí sola un entrenamiento estable.")
    heading("4. Optimización y selección")
    paragraph("Softmax transforma logits en probabilidades: p_k = exp(z_k) / sum_j exp(z_j). "
              "La función objetivo es J = -(1/N) sum_i log p(y_i|x_i) + "
              "lambda sum_kernels ||W||², con lambda = 0,0001.", "SmallCustom")
    paragraph("Se minimiza entropía cruzada categórica dispersa más penalización L2 de 0,0001 en "
              "los kernels convolucionales y la densa oculta. Adam usa tasa 0,001 y clipnorm=1: "
              "recorta la norma del gradiente de cada variable, no su norma global conjunta. "
              "Dropout 0,3 regulariza la capa oculta solo al entrenar. No se aplica aumento de datos.")
    paragraph(f"En cada fold, EarlyStopping observa val_loss (paciencia 3) y restaura los mejores pesos. "
              f"La mediana de las mejores épocas fue {result['final_epochs']}; con ese número fijo se "
              "reentrenó sobre todo desarrollo. La arquitectura se fijó antes del test; no hubo "
              "búsqueda de hiperparámetros ni ajuste de épocas usando sus resultados.")
    story.append(PageBreak())

    # Page 3: group validation and diagnostic comparisons.
    heading("5. Validación cruzada y sesgo-varianza")
    rows = [["Fold / grupo validado", "Acc. train", "Acc. val", "F1 macro val", "Brecha (pp)"]]
    gaps = []
    for fold in folds:
        gap = 100 * (fold["train"]["accuracy"] - fold["validation"]["accuracy"])
        gaps.append(gap)
        rows.append([f"{fold['fold']} / {','.join(map(str, fold['validation_groups']))}",
                     pct(fold["train"]["accuracy"]), pct(fold["validation"]["accuracy"]),
                     f"{fold['validation']['macro_f1']:.4f}", f"{gap:.2f}"])
    table(rows, [width * .29, width * .17, width * .17, width * .19, width * .18])
    paragraph(f"Accuracy de validación: media {pct(result['cv_accuracy_mean'])}, desviación estándar "
              f"muestral {100 * result['cv_accuracy_std']:.2f} puntos porcentuales (pp). "
              f"La brecha train-validación media es {mean(gaps):.2f} pp "
              f"(rango {min(gaps):.2f} a {max(gaps):.2f}). Se calcula con los pesos restaurados, "
              "en inferencia y sin dropout en ambas particiones; no se resta el accuracy de entrenamiento "
              "registrado durante actualizaciones del modelo.")
    plot("cv_learning_curves.png", width, 183)
    paragraph("Figura 1. Línea continua: entrenamiento; discontinua: validación. La pérdida de las curvas "
              "incluye L2, a diferencia de la entropía cruzada predictiva informada en las tablas.", "SmallCustom")
    baseline, half = (result["diagnostics"][key] for key in ("linear_baseline", "half_training_data"))
    full = folds[0]
    table([["Diagnóstico: mismo holdout", "n train", "Acc. train", "Acc. val"],
           ["Lineal: promedio 4 × 4 + softmax", baseline["train_count"], pct(baseline["train"]["accuracy"]), pct(baseline["validation"]["accuracy"])],
           ["CNN, mitad de datos", half["train_count"], pct(half["train"]["accuracy"]), pct(half["validation"]["accuracy"])],
           ["CNN, datos completos del fold", full["train_count"], pct(full["train"]["accuracy"]), pct(full["validation"]["accuracy"])]],
          [width * .49, width * .14, width * .18, width * .19])
    gain = 100 * (full["validation"]["accuracy"] - baseline["validation"]["accuracy"])
    data_gain = 100 * (full["validation"]["accuracy"] - half["validation"]["accuracy"])
    comparison = (f"La CNN supera al modelo lineal en {gain:.2f} pp" if gain >= 0 else
                  f"La CNN queda {abs(gain):.2f} pp por debajo del modelo lineal")
    paragraph(comparison + " de validación en ese holdout. " +
              f"Al pasar de la mitad a todos los datos del fold, la variación es {data_gain:+.2f} pp. "
              "Ambos diagnósticos usan la semilla del fold 1, el mismo holdout, máximo de épocas "
              "y criterio de parada; las épocas efectivas pueden diferir. El lineal cambia capacidad "
              "y representación. Con la mitad de datos hay menos actualizaciones por época: "
              "la comparación mezcla cantidad de datos y pasos de optimización.")
    training_error = 100 * mean(1 - fold["train"]["accuracy"] for fold in folds)
    validation_error = 100 * (1 - result["cv_accuracy_mean"])
    gap_interpretation = ("La mayor tasa de error fuera del entrenamiento es compatible con sobreajuste. "
                          if mean(gaps) > 0 else
                          "La brecha media no muestra una ventaja de entrenamiento en esta ejecución. ")
    budget_limit_count = sum(fold["best_epoch"] == config["epochs"] for fold in folds)
    paragraph(f"El error medio es {training_error:.2f}% en entrenamiento y {validation_error:.2f}% en "
              "validación. " + gap_interpretation +
              f"En {budget_limit_count}/5 folds, la mejor época coincide con el máximo de "
              f"{config['epochs']}: el presupuesto finito no demuestra convergencia. "
              "La optimización incompleta y el sobreajuste por muestra limitada pueden coexistir. "
              "La dispersión entre folds mezcla lote e inicialización; no descompone matemáticamente "
              "sesgo y varianza. Una corrida por condición no permite atribución causal: hacen falta "
              "semillas repetidas, más tamaños de muestra y grupos de procedencia reales.")
    story.append(PageBreak())

    # Page 4: true held-out results with a legible confusion matrix.
    heading("6. Evaluación final y análisis de errores")
    table([["Partición / agregación", "Accuracy", "F1 macro", "Entropía cruzada"],
           metric_row("Validación fuera de fold (OOF)", result["oof"]),
           metric_row("Desarrollo, modelo final", result["final_train"]),
           metric_row("Prueba retenida, modelo final", test)],
          [width * .43, width * .17, width * .17, width * .23])
    paragraph("Accuracy mide aciertos totales; F1 macro promedia por clase la media armónica de "
              "precisión y recall y pondera por igual las diez categorías. La entropía cruzada "
              "penaliza especialmente errores con alta confianza. OOF concatena predicciones "
              "de cinco modelos; prueba corresponde a un único modelo reentrenado.")
    plot("test_confusion_matrix.png", width, 347)
    paragraph("Figura 2. Matriz de confusión en prueba: filas = clase real, columnas = predicción; "
              "los valores son cantidades absolutas, no porcentajes.", "SmallCustom")
    matrix = test["confusion_matrix"]
    errors = sorted(((matrix[i][j], i, j) for i in range(10) for j in range(10) if i != j), reverse=True)[:3]
    paragraph("Las confusiones direccionales más frecuentes son " + "; ".join(
        f"{classes[i]} → {classes[j]} ({count} casos)" for count, i, j in errors) +
        ". Son errores observados; la matriz por sí sola no identifica qué rasgo visual causó cada uno.")
    reports = test["classification_report"]
    ranked = sorted(classes, key=lambda name: reports[name]["f1-score"])
    paragraph(f"La clase con mayor F1 es {ranked[-1]} ({reports[ranked[-1]]['f1-score']:.4f}); "
              f"la menor es {ranked[0]} ({reports[ranked[0]]['f1-score']:.4f}). "
              "El desempeño desigual aconseja revisar ejemplos por categoría antes de extender el "
              "modelo a imágenes ajenas a CIFAR-10. No demuestra que haya aprendido exclusivamente "
              "rasgos del objeto ni descarta atajos ligados a fondos u otros artefactos.")
    story.append(PageBreak())

    # Page 5: per-class metrics, perturbation, reproducibility and references.
    heading("7. Resultados por clase y robustez")
    rows = [["Clase", "Precisión", "Recall", "F1", "Soporte"]]
    for name in classes:
        entry = reports[name]
        rows.append([name, f"{entry['precision']:.3f}", f"{entry['recall']:.3f}",
                     f"{entry['f1-score']:.3f}", str(int(entry["support"]))])
    table(rows, [width * .32, width * .18, width * .17, width * .17, width * .16])
    shifted = result["robustness_shift_2px"]
    delta = 100 * (shifted["accuracy"] - test["accuracy"])
    paragraph(f"Al trasladar cada imagen de prueba 2 píxeles hacia abajo y a la derecha, la accuracy "
              f"es {pct(shifted['accuracy'])} ({delta:+.2f} pp respecto del original), "
              f"y F1 macro {shifted['macro_f1']:.4f}. Se rellenan con ceros el borde superior e izquierdo "
              "y se recortan dos filas y columnas del extremo opuesto. Por ello, el efecto combina "
              "traslación, pérdida de contenido y cambio de borde; no es una prueba aislada de invariancia.")
    paragraph("Compartir filtros favorece equivariancia local antes de bordes y submuestreo. El max pooling "
              "puede aportar tolerancia parcial a pequeños desplazamientos, pero stride 2 cambia el "
              "alineamiento de ventanas y no garantiza equivariancia exacta. Flatten y capas densas "
              "también dependen de la posición. Esta prueba no permite atribuir la respuesta solo al pooling.")
    heading("8. Reproducibilidad y conclusión")
    paragraph(f"Entorno registrado: Python {escape(result['versions']['python'])}, TensorFlow "
              f"{escape(result['versions']['tensorflow'])}, NumPy {escape(result['versions']['numpy'])}. "
              f"Se fijan semillas y operaciones deterministas. El modelo results/model.keras incorpora "
              f"normalización y acepta lotes RGB 32 × 32 en escala 0-255. Tras guardarlo y recargarlo, "
              f"la diferencia máxima de probabilidades sobre 32 imágenes fue {result['reload_max_abs_error']:.2g}.")
    paragraph(f"La ejecución alcanza {pct(test['accuracy'])} de accuracy en la prueba depurada. "
              "Las curvas, brechas, referencia lineal y reducción de datos sustentan un diagnóstico "
              "empírico limitado por el presupuesto y una sola semilla por condición. El repositorio "
              "incluye entrenamiento, inferencia, manifiesto, métricas y predicciones auditables. "
              "Para mejorar, corresponde evaluar aumento de datos y mayor presupuesto exclusivamente "
              "en desarrollo, reservando una nueva prueba si las decisiones usan estos resultados finales.")
    paragraph("Referencias", "SmallCustom")
    for text in [
        '[1] Krizhevsky, A. CIFAR-10: datos y descripción. <link href="https://www.cs.toronto.edu/~kriz/cifar.html" color="#087F8C">cs.toronto.edu/~kriz/cifar.html</link>',
        '[2] Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. <link href="https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf" color="#087F8C">Informe técnico</link>.',
        '[3] Keras. Conv2D layer. <link href="https://keras.io/api/layers/convolution_layers/convolution2d/" color="#087F8C">Documentación oficial de Conv2D</link>.',
        '[4] scikit-learn. GroupKFold. <link href="https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GroupKFold.html" color="#087F8C">Documentación oficial de GroupKFold</link>.',
    ]:
        paragraph(text, "SmallCustom")

    def decorate(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(teal)
        canvas.setLineWidth(.7)
        canvas.line(44, A4[1] - 31, A4[0] - 44, A4[1] - 31)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(navy)
        canvas.drawString(44, 25, "Inteligencia artificial | CNN sobre CIFAR-10")
        canvas.drawRightString(A4[0] - 44, 25, f"{document.page} / 5")
        canvas.restoreState()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(args.output), pagesize=A4, rightMargin=44, leftMargin=44,
                                 topMargin=44, bottomMargin=42,
                                 title="Trabajo práctico integrador - Parte 1: CNN", author="")
    document.build(story, onFirstPage=decorate, onLaterPages=decorate)
    from pypdf import PdfReader
    page_count = len(PdfReader(str(args.output)).pages)
    if page_count != 5:
        raise RuntimeError(f"Expected exactly 5 pages, got {page_count}; inspect and adjust layout.")
    print(f"Report created: {args.output.resolve()} ({page_count} pages)")


if __name__ == "__main__":
    main()

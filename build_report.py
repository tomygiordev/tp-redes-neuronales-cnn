"""Build a concise three-page Spanish report for the CNN part of the assignment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from xml.sax.saxutils import escape

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


from cnn import CLASS_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("report/informe_cnn.pdf"))
    args = parser.parse_args()

    result = json.loads((args.results / "results.json").read_text(encoding="utf-8"))
    runs, config, data = result["runs"], result["config"], result["dataset"]
    required = ("original_development_count", "excluded_track_ids", "excluded_groups",
                "excluded_images", "exact_overlap_images", "exclusion_rule", "integrity")
    if any(key not in data for key in required) or any(data["integrity"].values()):
        raise ValueError("Use the audited results with overlap exclusions and zero integrity conflicts.")
    main_run = next(run for run in runs if run["seed"] == result["main_seed"])
    baseline, test = result["baseline"], result["test"]

    navy = colors.HexColor("#17324D")
    teal = colors.HexColor("#087F8C")
    text_color = colors.HexColor("#263746")
    width = A4[0] - 80
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportTitle", fontName="Helvetica-Bold", fontSize=20, leading=23,
                              textColor=navy, spaceAfter=8))
    styles.add(ParagraphStyle("Section", fontName="Helvetica-Bold", fontSize=12.2, leading=15,
                              textColor=navy, spaceBefore=5, spaceAfter=6))
    styles.add(ParagraphStyle("Body", fontName="Helvetica", fontSize=9, leading=11.6,
                              textColor=text_color, spaceAfter=5))
    styles.add(ParagraphStyle("Small", fontName="Helvetica", fontSize=8, leading=10,
                              textColor=colors.HexColor("#4A5B68"), spaceAfter=4))
    styles.add(ParagraphStyle("Cell", fontName="Helvetica", fontSize=8, leading=10))
    story = []

    def paragraph(text, style="Body"):
        story.append(Paragraph(text, styles[style]))

    def heading(text):
        paragraph(text, "Section")

    def table(rows, widths):
        cells = [[Paragraph(escape(str(cell)), styles["Cell"]) for cell in row] for row in rows]
        item = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5EFF4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), navy),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ("LINEBELOW", (0, 0), (-1, 0), 0.7, teal),
            ("LINEBELOW", (0, 1), (-1, -1), 0.2, colors.HexColor("#D6E0E6")),
        ]))
        story.extend([item, Spacer(1, 5)])

    def plot(filename, max_height):
        path = args.results / filename
        image_width, image_height = ImageReader(str(path)).getSize()
        scale = min(width / image_width, max_height / image_height)
        story.append(Image(str(path), width=image_width * scale, height=image_height * scale))

    def pct(value):
        return f"{100 * value:.2f}".replace(".", ",") + " %"

    paragraph("TRABAJO PRÁCTICO INTEGRADOR - MÓDULO CNN", "Small")
    paragraph("Reconocimiento de señales con GTSRB", "ReportTitle")
    paragraph("<b>Objetivo.</b> Clasificar imágenes de señales alemanas en 43 categorías y evaluar la "
              "respuesta ante nuevas instancias mediante una CNN en Keras.")
    heading("1. Datos y prevención de fuga de información")
    paragraph("GTSRB permite trabajar con grupos reales: cada <i>track</i> reúne tomas de una misma "
              "señal física. Esta estructura permite separar instancias, en lugar de repartir "
              "fotogramas vecinos entre entrenamiento y validación. Se usan las 43 clases oficiales [1]. "
              "La unidad de separación es (clase, track), no el fotograma.")
    table([
        ["Partición", "Imágenes", "Criterio"],
        ["Entrenamiento", f"{data['train_count']:,}".replace(",", "."), f"{data['train_groups']} tracks"],
        ["Validación", f"{data['validation_count']:,}".replace(",", "."), f"{data['validation_groups']} tracks separados"],
        ["Test oficial", f"{data['test_count']:,}".replace(",", "."), "Partición oficial completa"],
    ], [width * .24, width * .18, width * .58])
    paragraph("Se reserva aproximadamente el 20% de los tracks de cada clase para validación con "
              "semilla 42. Las 43 clases están presentes y ningún track cruza train-validación. "
              "El manifiesto conserva identificadores, particiones y exclusiones. Esta división por "
              "grupos corresponde a la alternativa por lotes de la consigna.", "Small")
    paragraph(f"El control de duplicados detectó {data['exact_overlap_images']} imágenes de desarrollo "
              f"idénticas a imágenes del test. Se excluyó el grupo completo al que pertenecían "
              f"({data['excluded_images']} imágenes) después de fijar la partición. La validación y el test "
              "conservaron sus imágenes. La limpieza comparó píxeles, sin usar etiquetas ni rendimiento "
              "del test. Hubo una evaluación previa a esta corrección; se repitió el protocolo con los "
              "mismos hiperparámetros y semillas, sin ajustarlos a ese resultado.", "Small")
    paragraph("Preprocesamiento compartido: imagen completa suministrada, conversión RGB, "
              "redimensionado bilineal a 32 x 32 y división por 255 dentro del modelo. No se aplican "
              "aumentos, recortes adicionales ni estadísticas ajustadas con validación o test. La "
              "resolución reduce costo de cálculo, aunque puede perder detalles finos.", "Small")
    heading("2. Arquitectura y fundamento matemático")
    table([
        ["Operación", "Salida", "Parámetros"],
        ["Entrada + Rescaling", "32 x 32 x 3", "0"],
        ["Conv 3x3, 16 + MaxPool", "16 x 16 x 16", "448"],
        ["Conv 3x3, 32 + MaxPool", "8 x 8 x 32", "4.640"],
        ["Conv 3x3, 64 + MaxPool", "4 x 4 x 64", "18.496"],
        ["Flatten + Dense 64 + Dropout 0,3", "64", "65.600"],
        ["Dense 43 + Softmax", "43", "2.795"],
        ["Total entrenable", "", "91.979"],
    ], [width * .57, width * .24, width * .19])
    paragraph("<b>Convolución 2D.</b> z[i,j,f] = beta[f] + sum(u,v,c) W[u,v,c,f] x[i+u-p,j+v-p,c], "
              "con u,v = 0,1,2; c = 0,...,C-1 y p=1. C es la cantidad de canales de entrada, "
              "f identifica uno de los F filtros y beta[f] es su sesgo. Es la correlación cruzada de Conv2D. El filtro "
              "compartido detecta patrones locales con (3·3·C+1)·F parámetros. Stride 1 y padding "
              "<i>same</i> conservan la resolución: n_out = floor((n+2p-k)/s)+1 = n, "
              "con tamaño n, kernel k=3, padding p=1 y stride s=1 [2].", "Small")
    paragraph("<b>Pooling.</b> y[i,j,c] = max(a,b en {0,1}) x[2i+a,2j+b,c]. La ventana 2 x 2 con "
              "stride 2 reduce a la mitad cada dimensión. Puede tolerar desplazamientos locales que "
              "preserven el máximo; no garantiza invariancia ni equivarianza exacta ante cualquier "
              "traslación. Padding y submuestreo introducen efectos de borde y alineación [2,3].", "Small")
    paragraph("<b>Activaciones y pérdida.</b> ReLU(z)=max(0,z) aporta no linealidad sin saturación "
              "en el semieje positivo; He usa Var(W)=2/fan_in para conservar escala. Softmax: "
              "p_k=exp(z_k)/sum_j exp(z_j). Se minimiza L=-(1/N) sum_i log p(y_i|x_i) + "
              "0,0001 sum W². N es el tamaño del lote; W incluye kernels convolucionales y "
              "de Dense 64, sin sesgos ni capa de salida.", "Small")
    story.append(PageBreak())

    heading("3. Entrenamiento y decisiones de diseño")
    paragraph("Los tres bloques 16/32/64 aumentan canales mientras reducen resolución; Dense 64 "
              "mantiene pequeña la red. Adam con tasa 0,001 realiza actualizaciones adaptativas; "
              f"batch {config['batch_size']} equilibra memoria y costo. L2=0,0001 penaliza pesos grandes y Dropout=0,3 "
              "dificulta que la red dependa de combinaciones fijas de unidades. Estos valores se "
              "mantuvieron en la corrección del dataset; no se hizo búsqueda de hiperparámetros.")
    paragraph("El clipping por norma 1 limita cada tensor de gradiente como g'=g·min(1,1/||g||). "
              "Si ||g||=0, g'=0. Se mantuvo como precaución del diseño original, sin evidencia de "
              "que fuera necesario. "
              f"Se permiten {config['epochs']} épocas con early stopping de paciencia 5 sobre pérdida "
              "de validación, restaurando los mejores pesos. No se reentrena sobre validación.")
    paragraph("Las semillas 42, 43 y 44 cambian inicialización y orden de entrenamiento sobre la misma "
              "partición. Se fija de antemano la 42 para entregar; no se elige la mejor corrida. "
              "La referencia lineal recibe los mismos píxeles 32 x 32 y una única capa softmax "
              "(132.139 parámetros). Compara familias de modelos, sin aislar causalmente una operación.")
    heading("4. Ajuste, generalización y variabilidad")
    rows = [["Modelo / semilla", "Época elegida", "Acc. train", "Acc. val.", "F1 val."]]
    for label, run in [("Lineal / 42", baseline)] + [(f"CNN / {r['seed']}", r) for r in runs]:
        rows.append([label, run["best_epoch"], pct(run["train"]["accuracy"]),
                     pct(run["validation"]["accuracy"]), f"{run['validation']['macro_f1']:.4f}"])
    table(rows, [width * f for f in (.28, .18, .18, .18, .18)])
    plot("learning_curves.png", 195)
    paragraph("Figura 1. Entrenamiento continuo y validación discontinua, con un color por semilla. "
              "En entrenamiento actúa dropout y los pesos cambian durante la época; la tabla "
              "reevalúa los pesos restaurados con dropout desactivado. Por eso ambas accuracies "
              "de entrenamiento pueden diferir.", "Small")
    accuracies = [r["validation"]["accuracy"] for r in runs]
    gaps = [r["train"]["accuracy"] - r["validation"]["accuracy"] for r in runs]
    gain = main_run["validation"]["accuracy"] - baseline["validation"]["accuracy"]
    paragraph(f"La CNN promedia {pct(mean(accuracies))} de accuracy de validación, con desviación "
              f"muestral de {100 * stdev(accuracies):.2f} puntos porcentuales entre semillas. "
              f"La brecha media train-validación es {100 * mean(gaps):.2f} pp. La CNN principal "
              f"supera al modelo lineal en {100 * gain:.2f} pp.")
    paragraph(f"El error de entrenamiento de la CNN principal es {pct(1 - main_run['train']['accuracy'])}. "
              f"Su brecha con validación es {100 * (main_run['train']['accuracy'] - main_run['validation']['accuracy']):.2f} pp: "
              "logra ajustar el entrenamiento, pero conserva errores ante otras señales. Esto indica "
              "sobreajuste residual. La referencia lineal también muestra una brecha "
              f"({100 * (baseline['train']['accuracy'] - baseline['validation']['accuracy']):.2f} pp); "
              "su menor rendimiento no se explica sólo por sesgo alto. La comparación cambia arquitectura "
              "y regularización, por lo que no aísla el efecto de las convoluciones.")
    capped = sum(len(r["history"]["loss"]) == config["epochs"] for r in runs)
    paragraph(f"{capped} de las 3 CNN alcanzaron el máximo de {config['epochs']} épocas; no se puede "
              "asegurar que su pérdida haya convergido. La dispersión entre semillas mide sensibilidad "
              "al entrenamiento en esta partición, no variabilidad entre muestras ni una descomposición "
              "formal de sesgo y varianza. La validación se usa para elegir la época; el test evalúa "
              "el modelo principal fijado de antemano.", "Small")
    story.append(PageBreak())

    heading("5. Evaluación final y análisis de errores")
    table([
        ["CNN principal / semilla 42", "Accuracy", "F1 macro", "Entropía cruzada"],
        ["Entrenamiento", pct(main_run["train"]["accuracy"]), f"{main_run['train']['macro_f1']:.4f}", f"{main_run['train']['cross_entropy']:.4f}"],
        ["Validación", pct(main_run["validation"]["accuracy"]), f"{main_run['validation']['macro_f1']:.4f}", f"{main_run['validation']['cross_entropy']:.4f}"],
        ["Test oficial", pct(test["accuracy"]), f"{test['macro_f1']:.4f}", f"{test['cross_entropy']:.4f}"],
    ], [width * f for f in (.37, .17, .19, .27)])
    paragraph("Accuracy mide aciertos por imagen. Las clases tienen frecuencias diferentes: F1 macro "
              "promedia las 43 clases con el mismo peso; la pérdida no lleva pesos por clase. "
              "La entropía cruzada de la tabla excluye L2, mientras las curvas muestran pérdida "
              "regularizada. results.json conserva precisión, recall, F1 y soporte por clase.")
    reports = test["classification_report"]
    worst = sorted(range(len(CLASS_NAMES)), key=lambda i: reports[CLASS_NAMES[i]]["f1-score"])[:5]
    rows = [["Clases con menor F1 en test", "Soporte", "Recall", "F1"]]
    for index in worst:
        label, entry = CLASS_NAMES[index], reports[CLASS_NAMES[index]]
        rows.append([f"{index}: {label}", int(entry["support"]), pct(entry["recall"]), f"{entry['f1-score']:.4f}"])
    table(rows, [width * f for f in (.57, .13, .15, .15)])
    matrix = test["confusion_matrix"]
    errors = sorted(((matrix[i][j], i, j) for i in range(43) for j in range(43) if i != j), reverse=True)[:4]
    rows = [["Clase real > predicha (ID)", "Errores", "% de la clase real"]]
    for count, i, j in errors:
        rows.append([f"{i}: {CLASS_NAMES[i]} > {j}: {CLASS_NAMES[j]}", count, pct(count / sum(matrix[i]))])
    table(rows, [width * f for f in (.70, .12, .18)])
    paragraph("El porcentaje usa todas las imágenes de la clase real como denominador. La matriz "
              "completa de 43 x 43 está en test_confusion_matrix.png y results.json. El color expresa "
              "la proporción dentro de cada clase real; los números escritos son conteos. "
              "Las tablas muestran qué clases concentran los errores.", "Small")
    heading("6. Conclusiones y alcance")
    paragraph(f"La CNN de 91.979 parámetros alcanza {pct(test['accuracy'])} de accuracy y "
              f"{test['macro_f1']:.4f} de F1 macro en el test oficial. Separar señales físicas evita "
              "compartir fotogramas de una misma instancia entre entrenamiento y validación. "
              "La limpieza elimina coincidencias exactas con el test. Este control no descarta "
              "similitudes visuales ni todos los posibles atajos del dataset.")
    paragraph("Se estudia una partición y tres semillas, sin búsqueda de hiperparámetros. No se mide "
              "variabilidad entre particiones ni se garantiza desempeño en otros países, cámaras o "
              "escenas. El modelo clasifica recortes de señales; no detecta señales en fotografías "
              "completas. El archivo exportado incluye reescalado e infer.py repite el redimensionado.")
    heading("Referencias y reproducción")
    paragraph('[1] Stallkamp, J.; Schlipsing, M.; Salmen, J.; Igel, C. (2012). <i>Man vs. Computer: '
              'Benchmarking Machine Learning Algorithms for Traffic Sign Recognition.</i> Neural Networks, '
              '32, 323-332. <link href="https://christian-igel.github.io/paper/MvCBMLAfTSR.pdf" color="#087F8C">Artículo original</link>.', "Small")
    paragraph('[2] Keras: <link href="https://keras.io/api/layers/convolution_layers/convolution2d/" color="#087F8C">Conv2D</link>. '
              '[3] Keras: <link href="https://keras.io/api/layers/pooling_layers/max_pooling2d/" color="#087F8C">MaxPooling2D</link>. '
              'Datos: <link href="https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html" '
              'color="#087F8C">archivo oficial GTSRB</link>. Versiones, hashes y manifiesto acompañan '
              'los resultados; README.md detalla la reproducción.', "Small")

    def decorate(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D5E1E7"))
        canvas.line(40, 32, A4[0] - 40, 32)
        canvas.setFont("Helvetica", 7.3)
        canvas.setFillColor(navy)
        canvas.drawString(40, 20, "Redes Neuronales - CNN sobre GTSRB")
        canvas.drawRightString(A4[0] - 40, 20, f"Página {document.page} / 3")
        canvas.restoreState()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(args.output), pagesize=A4, rightMargin=40, leftMargin=40,
                                 topMargin=36, bottomMargin=42, title="Informe CNN - GTSRB")
    document.build(story, onFirstPage=decorate, onLaterPages=decorate)
    page_count = len(PdfReader(str(args.output)).pages)
    if page_count != 3:
        raise RuntimeError(f"Expected exactly 3 pages, got {page_count}; inspect the layout.")
    print(f"Report created: {args.output.resolve()} ({page_count} pages)")


if __name__ == "__main__":
    main()

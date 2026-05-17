# =========================================================
# RED NEURONAL MULTICLASE - CLASIFICACIÓN DE DÍGITOS
# KERAS + OPENCV + MNIST
#
# MODELO MATEMÁTICO:
#   Forward:  z = X·β + b  →  ReLU(z)  →  Softmax(z)
#   Pérdida:  L = -Σ y·log(ŷ)
#   Optimiz.: Adam (backpropagation automática)
# =========================================================

# =========================================================
# LIBRERÍAS
# =========================================================

import sys
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from tkinter import Tk
from tkinter.filedialog import askopenfilename

# TensorFlow / Keras — importaciones unificadas
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

from sklearn.metrics import confusion_matrix

# =========================================================
# CONFIGURACIÓN
# =========================================================

MODELO_PATH = "modelo_mnist.keras"

# Umbral de confianza para aceptar una predicción.
# 0.55 es más tolerante con tipografías de impresora/placa
# que difieren del estilo manuscrito de MNIST.
CONFIANZA_MINIMA = 0.55

# =========================================================
# CARGAR DATASET MNIST
# =========================================================

print("=" * 55)
print("  SISTEMA DE CLASIFICACIÓN DE DÍGITOS — MNIST")
print("=" * 55)
print("\nCargando dataset MNIST...")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Dataset cargado correctamente")

# =========================================================
# NORMALIZACIÓN: 0–255  →  0.0–1.0
# Divide cada pixel entre 255 para mantener valores
# en el rango [0,1], lo que estabiliza el entrenamiento.
# =========================================================

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# =========================================================
# CONVERSIÓN: imagen 28×28  →  vector de 784 características
#
# Cada imagen se "aplana" en un vector fila:
#   X ∈ ℝ^(n × 784)   donde n = número de muestras
# =========================================================

x_train = x_train.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)

# =========================================================
# INFORMACIÓN DEL DATASET
# =========================================================

print("\nInformación del dataset:")
print(f"  Entradas entrenamiento : {x_train.shape}")
print(f"  Etiquetas entrenamiento: {y_train.shape}")
print(f"  Entradas prueba        : {x_test.shape}")
print(f"  Etiquetas prueba       : {y_test.shape}")

# =========================================================
# CREAR O CARGAR MODELO
# =========================================================

if os.path.exists(MODELO_PATH):

    print(f"\nModelo encontrado en '{MODELO_PATH}'. Cargando...")
    model = keras.models.load_model(MODELO_PATH)
    print("Modelo cargado correctamente.")

else:

    print("\nNo se encontró modelo guardado. Creando red neuronal...")

    # =====================================================
    # ARQUITECTURA DE LA RED NEURONAL
    #
    # Entrada  →  Capa 1 (ReLU)  →  Capa 2 (ReLU)  →  Salida (Softmax)
    #
    # Cada capa densa calcula:
    #   z = X·β + b          (ecuación de la recta generalizada)
    #   a = ReLU(z)          (activación en capas ocultas)
    #   ŷ = Softmax(z)       (activación en capa de salida)
    #
    # Donde:
    #   β  = matriz de pesos  (shape: entradas × neuronas)
    #   b  = vector de bias   (shape: neuronas)
    #   X  = vector de entrada (shape: batch × 784)
    # =====================================================

    model = keras.Sequential([

        # Capa de entrada explícita — 784 características
        layers.Input(shape=(784,)),

        # -------------------------------------------------
        # CAPA OCULTA 1
        #   z₁ = X·β₁ + b₁         (784 → 128)
        #   a₁ = ReLU(z₁) = max(0, z₁)
        # -------------------------------------------------
        layers.Dense(
            128,
            activation='relu',
            use_bias=True,
            name="capa_oculta_1"
        ),

        # Dropout: desactiva 20 % de neuronas aleatoriamente
        # durante el entrenamiento para reducir overfitting
        layers.Dropout(0.2, name="dropout_1"),

        # -------------------------------------------------
        # CAPA OCULTA 2
        #   z₂ = a₁·β₂ + b₂        (128 → 64)
        #   a₂ = ReLU(z₂) = max(0, z₂)
        # -------------------------------------------------
        layers.Dense(
            64,
            activation='relu',
            use_bias=True,
            name="capa_oculta_2"
        ),

        # -------------------------------------------------
        # CAPA DE SALIDA
        #   z₃ = a₂·β₃ + b₃        (64 → 10)
        #   ŷᵢ = e^(zᵢ) / Σ e^(zⱼ)  (Softmax multiclase)
        # Produce una distribución de probabilidad sobre
        # los 10 dígitos posibles (0–9).
        # -------------------------------------------------
        layers.Dense(
            10,
            activation='softmax',
            use_bias=True,
            name="capa_salida"
        )
    ])

    # =====================================================
    # COMPILAR EL MODELO
    #
    # Función de pérdida: Entropía cruzada categórica
    #   L = -Σ yᵢ · log(ŷᵢ)
    #
    # Optimizador: Adam (variante adaptativa de SGD con
    #   momentum, estima primer y segundo momento del
    #   gradiente para actualizar pesos eficientemente)
    # =====================================================

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # =====================================================
    # RESUMEN DE LA RED
    # =====================================================

    print("\nResumen del modelo:\n")
    model.summary()

    # =====================================================
    # ENTRENAMIENTO
    #   - epochs     : número de pasadas completas sobre
    #                  el dataset
    #   - batch_size : muestras procesadas antes de
    #                  actualizar pesos (mini-batch SGD)
    #   - validation : evalúa en test al final de cada
    #                  época para detectar overfitting
    # =====================================================

    print("\nEntrenando red neuronal...\n")

    historial = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(x_test, y_test)
    )

    # =====================================================
    # EVALUACIÓN
    # =====================================================

    print("\nEvaluando modelo en conjunto de prueba...\n")

    loss, accuracy = model.evaluate(x_test, y_test)

    print(f"\nLoss    : {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}  ({accuracy*100:.2f} %)")

    # =====================================================
    # GUARDAR MODELO
    # =====================================================

    model.save(MODELO_PATH)
    print(f"\nModelo guardado en '{MODELO_PATH}'")

    # =====================================================
    # GRÁFICA DE ACCURACY
    # =====================================================

    plt.figure(figsize=(8, 4))

    plt.plot(historial.history['accuracy'],     label='Entrenamiento')
    plt.plot(historial.history['val_accuracy'], label='Validación')

    plt.title("Accuracy por época")
    plt.ylabel("Accuracy")
    plt.xlabel("Época")
    plt.legend()
    plt.tight_layout()
    plt.show()

# =========================================================
# MATRIZ DE CONFUSIÓN (Se calcula siempre)
#   Filas   = clase real
#   Columnas = clase predicha
# =========================================================

print("\nEvaluando datos de prueba para generar la matriz de confusión...")
predicciones = model.predict(x_test, verbose=0)
predicciones = np.argmax(predicciones, axis=1)

matriz = confusion_matrix(y_test, predicciones)

print("\nMatriz de confusión:\n")
print(matriz)

print("\nDetalle de aciertos y errores por dígito:")
print(f"{'Dígito':<8} | {'Verdaderos Pos. (TP)':<22} | {'Falsos Pos. (FP)':<18} | {'Falsos Neg. (FN)':<18} | {'Verdaderos Neg. (TN)':<22}")
print("-" * 98)
for i in range(10):
    TP = matriz[i, i]
    FP = np.sum(matriz[:, i]) - TP
    FN = np.sum(matriz[i, :]) - TP
    TN = np.sum(matriz) - (TP + FP + FN)
    print(f"{i:<8} | {TP:<22} | {FP:<18} | {FN:<18} | {TN:<22}")

# =========================================================
# SELECCIONAR IMAGEN EXTERNA
# =========================================================

print("\nSeleccione una imagen para clasificar...")

root = Tk()
root.withdraw()

ruta = askopenfilename(
    title="Seleccione una imagen con números",
    filetypes=[
        ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
        ("Todos los archivos", "*.*")
    ]
)

root.destroy()   # ← liberar ventana Tkinter correctamente

# =========================================================
# VALIDAR SELECCIÓN
# =========================================================

if not ruta:
    print("No se seleccionó ninguna imagen. Saliendo...")
    sys.exit(0)

# =========================================================
# LEER IMAGEN
# =========================================================

with open(ruta, "rb") as f:
    data = f.read()

img = cv2.imdecode(
    np.frombuffer(data, np.uint8),
    cv2.IMREAD_COLOR
)

if img is None:
    print("Error: no se pudo cargar la imagen.")
    sys.exit(1)

print(f"\nImagen cargada: {ruta}")
print(f"Dimensiones  : {img.shape[1]} × {img.shape[0]} px")

# =========================================================
# COPIA ORIGINAL (para dibujar resultados finales)
# =========================================================

original = img.copy()

# =========================================================
# FUNCIÓN: MEJOR THRESHOLD AUTOMÁTICO
#
# Prueba dos métodos de binarización y elige el que
# produce más contornos candidatos de dígitos, adaptándose
# al contraste y tipo de imagen entregada.
#
# Método A — Adaptativo Gaussiano:
#   Compara cada píxel contra la media ponderada de su
#   vecindad local. Ideal para iluminación desigual.
#
# Método B — Otsu global:
#   Calcula el umbral óptimo que minimiza la varianza
#   intra-clase. Ideal para imágenes de alto contraste
#   (placas, documentos impresos).
# =========================================================

def mejor_threshold(gray_img):
    """
    Recibe imagen en escala de grises.
    Devuelve la binarización que contiene más contornos
    del tamaño apropiado para dígitos.
    """
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Método A — Threshold adaptativo
    th_adapt = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # Método B — Otsu (threshold global óptimo)
    _, th_otsu = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Contar contornos candidatos en cada método
    # Un candidato válido tiene área entre 0.1% y 20%
    # del área total de la imagen (proporcional)
    area_img = gray_img.shape[0] * gray_img.shape[1]
    area_min = area_img * 0.001
    area_max = area_img * 0.20

    def contar_candidatos(thresh_img):
        cnts, _ = cv2.findContours(
            thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        count = 0
        for c in cnts:
            a = cv2.contourArea(c)
            if area_min < a < area_max:
                _, _, cw, ch = cv2.boundingRect(c)
                ratio = cw / float(ch) if ch > 0 else 0
                if 0.08 < ratio < 3.0:
                    count += 1
        return count

    n_adapt = contar_candidatos(th_adapt)
    n_otsu  = contar_candidatos(th_otsu)

    print(f"  Threshold adaptativo → {n_adapt} candidatos")
    print(f"  Threshold Otsu       → {n_otsu} candidatos")

    if n_otsu >= n_adapt:
        print("  Seleccionado: Otsu")
        return th_otsu
    else:
        print("  Seleccionado: Adaptativo")
        return th_adapt


# =========================================================
# FUNCIÓN: DETECTAR ZONA DE INTERÉS AUTOMÁTICAMENTE
#
# Busca el rectángulo más grande con proporción de placa
# o panel de números dentro de la imagen. Si lo encuentra,
# recorta esa zona para eliminar el fondo irrelevante.
#
# Criterios de una zona válida:
#   - Ocupa entre 5% y 80% del área total
#   - Relación ancho/alto entre 1.0 y 6.0
#   - No es el rectángulo de la imagen entera
# =========================================================

def detectar_zona_interes(img_color):
    """
    Devuelve (zona_recortada, offset_x, offset_y).
    Si no encuentra zona específica, devuelve imagen completa
    con offsets en 0.
    """
    alt, anc = img_color.shape[:2]
    area_total = alt * anc

    gray_z = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    blur_z = cv2.GaussianBlur(gray_z, (5, 5), 0)

    # Usar Canny para detectar bordes fuertes (marcos, bordes de placa)
    edges = cv2.Canny(blur_z, 30, 150)

    # Dilatar para cerrar gaps en el contorno del marco
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges  = cv2.dilate(edges, kernel, iterations=2)

    cnts, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    mejor     = None
    mejor_area = 0

    for c in cnts:
        area_c = cv2.contourArea(c)

        # Ignorar contornos muy pequeños o que sean casi la imagen entera
        if area_c < area_total * 0.05:
            continue
        if area_c > area_total * 0.80:
            continue

        zx, zy, zw, zh = cv2.boundingRect(c)

        # Proporción rectangular válida (panel de números)
        ratio_z = zw / float(zh) if zh > 0 else 0
        if ratio_z < 1.0 or ratio_z > 6.0:
            continue

        if area_c > mejor_area:
            mejor_area = area_c
            mejor = (zx, zy, zw, zh)

    if mejor is not None:
        zx, zy, zw, zh = mejor
        # Pequeño margen de seguridad para no cortar bordes del dígito
        margen = int(min(zw, zh) * 0.03)
        zx = max(0, zx - margen)
        zy = max(0, zy - margen)
        zw = min(anc - zx, zw + margen * 2)
        zh = min(alt - zy, zh + margen * 2)

        zona = img_color[zy:zy+zh, zx:zx+zw]
        print(f"  Zona detectada automáticamente: "
              f"x={zx}, y={zy}, {zw}×{zh} px")
        return zona, zx, zy
    else:
        print("  No se detectó zona específica → usando imagen completa")
        return img_color, 0, 0


# =========================================================
# PASO 1 — DETECTAR ZONA DE INTERÉS
# =========================================================

print("\nAnalizando imagen...")
zona, offset_x, offset_y = detectar_zona_interes(img)

# =========================================================
# PASO 2 — ESCALA DE GRISES + MEJOR THRESHOLD
# =========================================================

print("\nSeleccionando método de threshold...")
gray  = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
thresh = mejor_threshold(gray)

# =========================================================
# PASO 3 — MORFOLOGÍA PARA SEPARAR DÍGITOS FUSIONADOS
#
# Erosión leve: reduce píxeles blancos en los bordes,
# separando dígitos que quedaron pegados tras el threshold.
# Se aplica solo si la imagen es pequeña (alta densidad
# de píxeles por dígito).
# =========================================================

alt_zona, anc_zona = zona.shape[:2]

if alt_zona < 150:
    # Imagen pequeña → erosión mínima (1 px)
    k_erosion = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
else:
    # Imagen más grande → erosión de 2 px
    k_erosion = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

thresh = cv2.erode(thresh, k_erosion, iterations=1)

# =========================================================
# PASO 4 — DETECCIÓN DE CONTORNOS
#   RETR_EXTERNAL : solo contornos exteriores
#   CHAIN_APPROX_SIMPLE : comprime segmentos redundantes
# =========================================================

contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Ordenar de izquierda a derecha por coordenada X
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

# =========================================================
# PASO 5 — FILTROS PROPORCIONALES AL TAMAÑO DE LA ZONA
#
# En lugar de usar píxeles fijos, todos los umbrales
# se calculan como porcentaje del área/dimensión de la
# zona detectada. Así el sistema funciona igual con una
# imagen de 100 px que con una de 3000 px.
# =========================================================

area_zona     = alt_zona * anc_zona
area_min_cont = area_zona * 0.001   # mínimo 0.1% del área
area_max_cont = area_zona * 0.18    # máximo 18% del área

# Altura mínima y máxima de un dígito (proporcional)
h_min = int(alt_zona * 0.10)   # al menos 10% de la altura de la zona
h_max = int(alt_zona * 0.95)   # no más del 95% de la altura de la zona

# Ancho máximo de un dígito (no puede ser más ancho que
# el 40% del ancho total → evita rectángulos de fondo)
w_max = int(anc_zona * 0.40)

print(f"\nFiltros adaptativos calculados:")
print(f"  Área contorno : {area_min_cont:.0f} – {area_max_cont:.0f} px²")
print(f"  Altura dígito : {h_min} – {h_max} px")
print(f"  Ancho máximo  : {w_max} px")

# =========================================================
# PASO 6 — SEGMENTACIÓN Y CLASIFICACIÓN
# =========================================================

numeros_detectados = []   # Lista de (x_global, dígito, confianza)

for contour in contours:

    area = cv2.contourArea(contour)

    # ----- Filtro de área proporcional -----
    if area < area_min_cont or area > area_max_cont:
        continue

    x, y, w, h = cv2.boundingRect(contour)

    # ----- Filtros de tamaño proporcionales -----
    if h < h_min or h > h_max:
        continue

    if w > w_max:
        continue

    # ----- Relación de aspecto -----
    # 0.08–3.0 cubre desde el "1" muy delgado
    # hasta dígitos anchos en cualquier tipografía
    ratio = w / float(h)

    if ratio < 0.08 or ratio > 3.0:
        continue

    # ----- Solidez proporcional -----
    # (área real / área del bounding box)
    solidity = area / float(w * h)

    if solidity < 0.15:
        continue

    # =====================================================
    # EXTRAER ROI Y CONVERTIR AL FORMATO MNIST
    # =====================================================

    roi = thresh[y:y+h, x:x+w]

    # Padding proporcional al tamaño del dígito detectado
    # (no un valor fijo de 20 px)
    padding = max(4, int(max(w, h) * 0.25))

    roi = cv2.copyMakeBorder(
        roi,
        padding, padding, padding, padding,
        cv2.BORDER_CONSTANT,
        value=0
    )

    # Redimensionar a 28×28 (formato MNIST)
    roi = cv2.resize(roi, (28, 28))

    # Asegurar que el dígito sea blanco sobre fondo negro
    # (igual que en MNIST). Si la media > 127 → invertir.
    if np.mean(roi) > 127:
        roi = 255 - roi

    # =====================================================
    # PREPARAR VECTOR DE ENTRADA
    #   Normalización: 0–255 → 0.0–1.0
    #   Reshape: (28,28) → vector (1, 784)
    # =====================================================

    roi_norm = roi.astype("float32") / 255.0
    roi_vec  = roi_norm.reshape(1, 784)

    # =====================================================
    # PREDICCIÓN — Forward Propagation
    #   z₁ = roi_vec · β₁ + b₁  →  ReLU(z₁)
    #   z₂ = a₁     · β₂ + b₂  →  ReLU(z₂)
    #   z₃ = a₂     · β₃ + b₃  →  Softmax(z₃) → ŷ
    # =====================================================

    pred = model.predict(roi_vec, verbose=0)

    numero    = int(np.argmax(pred))
    confianza = float(np.max(pred))

    # Descartar predicciones con baja confianza
    if confianza < CONFIANZA_MINIMA:
        continue

    # Coordenadas globales sobre la imagen original
    # (compensar el recorte de la zona de interés)
    x_global = x + offset_x
    y_global = y + offset_y

    numeros_detectados.append((x_global, numero, confianza))

    # ----- Dibujar rectángulo y etiqueta -----
    cv2.rectangle(
        original,
        (x_global, y_global),
        (x_global + w, y_global + h),
        (0, 255, 0),
        2
    )

    etiqueta = f"{numero}  ({confianza*100:.1f}%)"

    cv2.putText(
        original,
        etiqueta,
        (x_global, y_global - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

# =========================================================
# RESULTADO FINAL
# =========================================================

print("\n" + "=" * 40)

if numeros_detectados:

    # Ordenar por posición X (izquierda → derecha)
    numeros_detectados.sort(key=lambda item: item[0])

    secuencia = ''.join(str(num) for _, num, _ in numeros_detectados)

    print(f"Número(s) detectado(s): {secuencia}")
    print("\nDetalle por dígito:")

    for pos, (_, num, conf) in enumerate(numeros_detectados):
        print(f"  Posición {pos+1}: {num}  |  Confianza: {conf*100:.1f} %")

else:
    print("No se detectaron números con confianza suficiente.")
    print(f"(Umbral actual: {CONFIANZA_MINIMA*100:.0f} %)")

print("=" * 40)

# =========================================================
# MOSTRAR RESULTADOS EN PANTALLA
# =========================================================

cv2.imshow("Resultado Final — Detecciones", original)
cv2.imshow("Threshold (zona procesada)",    thresh)

print("\nPresione cualquier tecla sobre la ventana para cerrar...")
cv2.waitKey(0)
cv2.destroyAllWindows()
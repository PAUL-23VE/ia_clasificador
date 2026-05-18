# =========================================================
# RED NEURONAL MULTICLASE - CLASIFICACIÓN DE DÍGITOS
# KERAS + OPENCV + MNIST
# =========================================================

# --- IMPORTACIÓN DE LIBRERÍAS ---
# sys, os: Para interactuar con el sistema operativo (rutas, salir del programa).
import sys, os
# cv2: Librería OpenCV para todo el procesamiento de visión computacional (recortes, filtros, contornos).
# numpy as np: Librería fundamental para operaciones matemáticas avanzadas y manejo de matrices/vectores.
import cv2, numpy as np
# Tkinter: Usado exclusivamente para abrir la ventana nativa de selección de archivos de Windows/Linux.
from tkinter import Tk
from tkinter.filedialog import askopenfilename
# keras / tensorflow: El motor de Inteligencia Artificial. Keras es la API de alto nivel para construir la red.
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
# confusion_matrix: Herramienta matemática de scikit-learn para evaluar qué tanto se equivoca el modelo.
from sklearn.metrics import confusion_matrix

# =========================================================
# CONSTRAINT PARA RECORTAR PESOS (REGULARIZACIÓN MATEMÁTICA)
# =========================================================
class ClipWeights(keras.constraints.Constraint):
    """
    Constraint personalizado (Restricción Matemática).
    Defensa: En las redes neuronales, si los "pesos" (weights) crecen demasiado, 
    el modelo se vuelve inestable (fenómeno de gradientes explosivos) y memoriza en lugar de aprender.
    Esta clase fuerza a que, después de cada actualización matemática, ningún peso sea 
    menor a -1 ni mayor a 1.
    """
    def __init__(self, min_value=-1.0, max_value=1.0):
        self.min, self.max = float(min_value), float(max_value)
        
    def __call__(self, w):
        # tf.clip_by_value: Función de tensorFlow que recorta literalmente la matriz de pesos al rango dado.
        return tf.clip_by_value(w, self.min, self.max)
        
    def get_config(self):
        # Necesario para que Keras sepa cómo guardar esta clase personalizada en el archivo .keras
        return {'min_value': self.min, 'max_value': self.max}

# Instanciamos la restricción para usarla en todas las capas.
clip = ClipWeights(-1.0, 1.0)

# =========================================================
# CAPA DENSE PERSONALIZADA CON UN SOLO BIAS ESCALAR
# =========================================================
class OneBiasDense(keras.layers.Layer):
    """
    Defensa: Una capa "Dense" normal calcula Z = W*X + b, donde 'b' es un vector 
    (un sesgo por cada neurona). Esta clase matemática personalizada modifica esa 
    ecuación para que todas las neuronas de la capa compartan un ÚNICO sesgo escalar.
    Esto reduce el número de parámetros libres y obliga a la red a ser más eficiente.
    """
    def __init__(self, units, activation=None, kernel_constraint=None, bias_constraint=None, use_bias=True, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = int(units) # Número de neuronas en esta capa
        self.activation = keras.activations.get(activation) # Función de activación (ReLU o Softmax)
        self.kernel_constraint = keras.constraints.get(kernel_constraint) if kernel_constraint else None
        self.bias_constraint = keras.constraints.get(bias_constraint) if bias_constraint else None
        self.use_bias = use_bias
        # Creamos una capa densa normal pero le APAGAMOS el bias nativo (use_bias=False)
        self._dense = layers.Dense(self.units, use_bias=False, kernel_constraint=self.kernel_constraint)

    def build(self, input_shape):
        # Se ejecuta la primera vez que la capa recibe datos para inicializar variables
        self._dense.build(input_shape)
        if self.use_bias:
            # Aquí inyectamos matemáticamente NUESTRO bias: un escalar de forma (1,) inicializado en cero.
            self.bias = self.add_weight(shape=(1,), initializer='zeros', trainable=True, name='bias', constraint=self.bias_constraint)
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, inputs):
        # Forward Propagation de esta capa: Z = W*X
        x = self._dense(inputs)
        if self.bias is not None:
            # Z = (W*X) + b_escalar (gracias a las reglas de broadcasting de matrices)
            x = x + self.bias
        # Retorna Activación(Z)
        return self.activation(x) if self.activation else x

    def get_config(self):
        # Permite serializar y guardar nuestra capa extraña en el disco duro.
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": keras.activations.serialize(self.activation),
            "kernel_constraint": keras.constraints.serialize(self.kernel_constraint),
            "bias_constraint": keras.constraints.serialize(self.bias_constraint),
            "use_bias": self.use_bias,
        })
        return config

# =========================================================
# CONFIGURACIÓN GLOBAL
# =========================================================
MODELO_PATH = "modelo_mnist.keras" # Archivo donde vivirá la IA
# Umbral mínimo para aceptar un dígito en la vida real. 
# Si la red duda y predice con menos de 40% de certeza, ignoramos la predicción por seguridad.
CONFIANZA_MINIMA = 0.40

# =========================================================
# 1. CARGA Y PREPROCESAMIENTO DEL DATASET MNIST
# =========================================================
print("="*55)
print("  SISTEMA DE CLASIFICACIÓN DE DÍGITOS — MNIST")
print("="*55)
print("\nCargando dataset MNIST...")

# MNIST contiene 70,000 imágenes (60k entrenamiento, 10k prueba) de dígitos escritos a mano.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Dataset cargado correctamente")

# NORMALIZACIÓN MATEMÁTICA: 
# Los píxeles van de 0 (negro) a 255 (blanco). Dividimos entre 255.0 para aplastarlos al rango [0.0, 1.0].
# Defensa: Las redes neuronales trabajan infinitamente mejor y convergen más rápido con números pequeños entre 0 y 1.
x_train, x_test = x_train.astype("float32")/255.0, x_test.astype("float32")/255.0

# APLANAMIENTO (FLATTEN):
# Una imagen es una matriz 2D de 28x28 píxeles. 
# Nuestra red es un Perceptrón Multicapa (MLP) que requiere vectores 1D.
# Transformamos 28x28 en un vector fila de 784 posiciones matemáticas (28 * 28 = 784).
x_train, x_test = x_train.reshape(-1, 784), x_test.reshape(-1, 784)

print("\nInformación del dataset:")
print(f"  Entradas entrenamiento : {x_train.shape}")
print(f"  Etiquetas entrenamiento: {y_train.shape}")
print(f"  Entradas prueba        : {x_test.shape}")
print(f"  Etiquetas prueba       : {y_test.shape}")

# =========================================================
# 2. CARGAR MODELO EXISTENTE O CREAR UNO NUEVO
# =========================================================
if os.path.exists(MODELO_PATH):
    # Si ya entrenamos antes, no perdemos tiempo y lo cargamos directamente de memoria.
    print(f"\nModelo encontrado en '{MODELO_PATH}'. Cargando...")
    # Le pasamos custom_objects para que sepa cómo leer nuestras clases matemáticas raras.
    model = keras.models.load_model(MODELO_PATH, custom_objects={'ClipWeights': ClipWeights, 'OneBiasDense': OneBiasDense})
    print("Modelo cargado correctamente.")
    print("Recortando pesos cargados al rango [-1, 1]...")
    # Aseguramos por seguridad que los pesos al cargar sigan cumpliendo la restricción [-1, 1]
    for layer in model.layers:
        try:
            ws = layer.get_weights()
            if not ws: continue
            ws_clipped = [np.clip(w, -1.0, 1.0) for w in ws]
            layer.set_weights(ws_clipped)
        except Exception:
            pass
else:
    # Si borramos el archivo, armamos la arquitectura desde cero.
    print("\nNo se encontró modelo guardado. Creando red neuronal...")
    
    # ARQUITECTURA DEL PERCEPTRÓN MULTICAPA (MLP)
    model = keras.Sequential([
        # Capa de Entrada: 784 píxeles simultáneos.
        layers.Input(shape=(784,)),
        
        # Capa Oculta 1: 128 neuronas. 
        # Activación ReLU: f(x) = max(0, x). Permite aprender relaciones no lineales (curvas).
        OneBiasDense(128, activation='relu', kernel_constraint=clip, bias_constraint=clip, name="capa_oculta_1"),
        
        # Dropout (Regularización): 
        # Defensa: Apaga al azar el 30% de las conexiones en cada paso. Esto IMPIDE que la red "memorice" 
        # las imágenes exactas y la obliga a "aprender" los patrones generales, mejorando la robustez.
        layers.Dropout(0.3),  
        
        # Capa Oculta 2: 64 neuronas. Embudamos la información.
        OneBiasDense(64, activation='relu', kernel_constraint=clip, bias_constraint=clip, name="capa_oculta_2"),
        layers.Dropout(0.2),  # Apaga el 20% para seguir forzando el aprendizaje real.
        
        # Capa de Salida: 10 neuronas (dígitos del 0 al 9).
        # Activación Softmax: Convierte las salidas matemáticas brutas en Probabilidades (Porcentajes) que suman 100%.
        OneBiasDense(10, activation='softmax', kernel_constraint=clip, bias_constraint=clip, name="capa_salida")
    ])
    
    # COMPILACIÓN DEL MODELO
    # Optimizador Adam: Ajusta la "velocidad de aprendizaje" de forma dinámica (momentos de primer y segundo orden).
    # Función de Pérdida (Loss): Entropía Cruzada Categórica, ideal para clasificaciones multiclase.
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("\nEntrenando red neuronal...\n")
    # ENTRENAMIENTO (Backpropagation):
    # epochs=12: Repite el aprendizaje sobre todos los datos 12 veces.
    # batch_size=32: Lee de 32 en 32 imágenes antes de corregir sus errores, logrando estabilidad matemática.
    historial = model.fit(x_train, y_train, epochs=12, batch_size=32, validation_data=(x_test, y_test))
    model.save(MODELO_PATH) # Guardamos la red en disco.

# =========================================================
# 3. EVALUACIÓN DE SOBREAJUSTE (APRENDIZAJE VS MEMORIZACIÓN)
# =========================================================
print("\nCalculando errores de Entrenamiento y Pruebas...")
# Defensa: Evaluamos matemáticamente qué tan mal se equivoca con datos que conoce (train)
# frente a datos que NUNCA ha visto (test).
loss_train, acc_train = model.evaluate(x_train, y_train, verbose=0)
loss_test, acc_test   = model.evaluate(x_test, y_test, verbose=0)

print("\n" + "="*55)
print(" COMPARATIVA DE ERRORES (Pérdida y Precisión)")
print("="*55)
print(f"  Entrenamiento -> Error: {loss_train:.4f} | Precisión: {acc_train*100:.2f}%")
print(f"  Pruebas       -> Error: {loss_test:.4f} | Precisión: {acc_test*100:.2f}%")
# Si el modelo solo memorizara, Pruebas tendría un error gigante. Como son parecidos, APRENDIÓ.
print("="*55)

# =========================================================
# 4. MATRIZ DE CONFUSIÓN (EVALUACIÓN DETALLADA)
# =========================================================
print("\nEvaluando datos de prueba para generar la matriz de confusión...")
# Hacemos que la IA adivine las 10,000 imágenes de prueba
predicciones = np.argmax(model.predict(x_test, verbose=0), axis=1)
# Comparamos lo que predijo vs la realidad (y_test)
matriz = confusion_matrix(y_test, predicciones)

print("\nMatriz de confusión:\n")
print(matriz)

# Desglosamos la matriz de confusión matemáticamente para cada clase (0-9)
print("\nDetalle de aciertos y errores por dígito:")
print(f"{'Dígito':<8} | {'Verdaderos Pos. (TP)':<22} | {'Falsos Pos. (FP)':<18} | {'Falsos Neg. (FN)':<18} | {'Verdaderos Neg. (TN)':<22}")
print("-"*98)
for i in range(10):
    TP = matriz[i, i]                            # Predijo la clase i correctamente.
    FP = np.sum(matriz[:, i]) - TP               # Predijo clase i, pero en realidad era otra.
    FN = np.sum(matriz[i, :]) - TP               # Era clase i, pero el modelo predijo otra cosa.
    TN = np.sum(matriz) - (TP + FP + FN)         # Rechazó correctamente que no era la clase i.
    print(f"{i:<8} | {TP:<22} | {FP:<18} | {FN:<18} | {TN:<22}")

# =========================================================
# 5. CARGA DE IMAGEN REAL (INTERFAZ Y OPENCV)
# =========================================================
print("\nSeleccione una imagen para clasificar...")
root = Tk(); root.withdraw() # Ocultamos la ventana principal de Tkinter, solo queremos el diálogo de archivo.
ruta = askopenfilename(title="Seleccione una imagen con números", filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
root.destroy()
if not ruta: sys.exit(0) # Si cancela la ventana, salimos.

# imdecode: Carga segura mediante búfer binario (evita errores con rutas que contengan tildes o espacios).
img = cv2.imdecode(np.frombuffer(open(ruta,"rb").read(), np.uint8), cv2.IMREAD_COLOR)
if img is None: sys.exit(1)
print(f"\nImagen cargada: {ruta}")
print(f"Dimensiones  : {img.shape[1]} × {img.shape[0]} px")

original = img.copy() # Guardamos copia para dibujar cuadritos verdes/rojos encima al final.

# =========================================================
# 6. PREPROCESAMIENTO OPENCV (VISIÓN COMPUTACIONAL)
# =========================================================
print("\nAnalizando imagen...")
print("  No se detectó zona específica → usando imagen completa")

# Convertir a escala de grises para eliminar ruido de color.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Filtro Gaussiano: Difumina levemente la imagen para que el ruido o polvo desaparezca.
blur = cv2.GaussianBlur(gray,(5,5),0)

# Binarización de Otsu: 
# Defensa: Calcula matemáticamente el valle óptimo en el histograma de colores para separar 
# el "fondo" negro de la "tinta" blanca.
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Operación Morfológica "Close" (Cierre):
# Defensa: Si un trazo tiene huecos microscópicos por mala calidad de cámara, 
# esto dilata y erosiona para "cerrar" esos huecos y unificar el número.
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

print("\nBinarización (Otsu) y Morfología aplicadas exitosamente.")

# Extraemos los "contornos" (los perímetros de los posibles números)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Ordenamos matemáticamente de izquierda a derecha usando su coordenada X inicial
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

# =========================================================
# 7. FILTROS ADAPTATIVOS
# =========================================================
# Defensa: En lugar de buscar un número de píxeles estricto, calculamos umbrales 
# dinámicos basados en la resolución de LA IMAGEN cargada (1.5% del total, 15% altura, etc).
area_img = img.shape[0]*img.shape[1]
area_min, area_max = area_img*0.015, area_img
h_min, h_max = int(img.shape[0]*0.15), img.shape[0]
w_max = img.shape[1]

print("\nFiltros adaptativos calculados:")
print(f"  Área contorno : {int(area_min)} – {int(area_max)} px²")
print(f"  Altura dígito : {h_min} – {h_max} px")
print(f"  Ancho máximo  : {w_max} px")

# =========================================================
# 8. EXTRACCIÓN DE ROI (REGIONES DE INTERÉS) Y PREDICCIÓN
# =========================================================
numeros_detectados = []   # Guardará tuplas: (posición X, dígito adivinado, confianza)

for contour in contours:
    # Bounding Rect: Crea una "caja" matemática (X, Y, Ancho, Alto) perfecta que encierra el contorno.
    x,y,w,h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)

    # --- Aplicación de los Filtros ---
    if area < area_min or area > area_max:
        print(f"  [Filtro] Contorno (X:{x}, Y:{y}) descartado por Área: {area} (Min:{area_min}, Max:{area_max})")
        continue

    if h < h_min or h > h_max:
        print(f"  [Filtro] Contorno (X:{x}, Y:{y}) descartado por Altura: {h} (Min:{h_min}, Max:{h_max})")
        continue
    if w > w_max:
        print(f"  [Filtro] Contorno (X:{x}, Y:{y}) descartado por Ancho: {w} (Max:{w_max})")
        continue

    # Filtro Ratio: Ancho / Alto. Si es menor a 0.08 o mayor a 3.0, es una línea errónea, no un número.
    ratio = w / float(h)
    if ratio < 0.08 or ratio > 3.0:
        print(f"  [Filtro] Contorno (X:{x}, Y:{y}) descartado por Proporción (Ratio): {ratio:.2f}")
        continue

    # Filtro Solidez: Relación entre área entintada vs área de la caja. Si es muy bajo, es ruido.
    solidity = area / float(w * h)
    if solidity < 0.10:
        print(f"  [Filtro] Contorno (X:{x}, Y:{y}) descartado por Solidez: {solidity:.2f}")
        continue

    # =====================================================
    # TRANSFORMAR LA IMAGEN REAL AL FORMATO EXACTO DE MNIST
    # =====================================================
    # Recortamos el pedazo de imagen que nos interesa (ROI)
    roi = thresh[y:y+h, x:x+w]

    # PASO 1: Forzar a que la imagen recortada sea un CUADRADO PERFECTO agregando bordes negros.
    # Defensa: Las imágenes sin deformar la relación de aspecto funcionan infinitamente mejor en IA.
    diff = abs(w - h)
    top, bottom, left, right = 0, 0, 0, 0
    if w > h:
        top = diff // 2
        bottom = diff - top
    else:
        left = diff // 2
        right = diff - left
    roi_cuadrado = cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    # PASO 1.5: Dilatación
    # Defensa: Ayuda a solucionar el problema de trazos finos (como un 5 engañoso que se parece a 3).
    # Engrosa los trazos blancos un píxel para igualar el grosor de los dígitos del laboratorio MNIST.
    kernel_dilatacion = np.ones((2,2), np.uint8)
    roi_cuadrado = cv2.dilate(roi_cuadrado, kernel_dilatacion, iterations=1)

    # PASO 2: Añadir padding (Borde protector) 
    # Defensa: En MNIST, los números nunca tocan los bordes, están aislados en el centro (al 20%).
    pad = int(roi_cuadrado.shape[0] * 0.18)
    roi_padded = cv2.copyMakeBorder(roi_cuadrado, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    # PASO 3: Redimensionar matemáticamente a la cuadrícula sagrada de 28x28 píxeles.
    roi = cv2.resize(roi_padded, (28, 28), interpolation=cv2.INTER_AREA)

    # PASO 4: Centrado por Momentos Espaciales (CENTRO DE MASA)
    # Defensa (¡CRUCIAL PARA DEFENDER!): Los perceptrones multicapa son sensibles a si el número
    # está desplazado. Esta función calcula el peso visual en los ejes X,Y (M["m10"], M["m01"])
    # y averigua matemáticamente cuántos píxeles debemos mover la tinta para que su centro de gravedad
    # quede anclado perfectamente en el centro geométrico (14,14). Así generaron MNIST originalmente Yann LeCun.
    M = cv2.moments(roi)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        # Ecuación de traslación afín:
        shift_x = 14.0 - cx
        shift_y = 14.0 - cy
        M_warp = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        roi = cv2.warpAffine(roi, M_warp, (28, 28))

    # Aplanar y normalizar [0,1] la imagen extraída del mundo real para la IA
    roi_norm = roi.astype("float32") / 255.0
    roi_vec  = roi_norm.reshape(1, 784)

    # =====================================================
    # 9. PREDICCIÓN MATEMÁTICA CON LA IA
    # =====================================================
    # Forward Propagation final. Retorna las probabilidades (Softmax) de que sea cada número del 0 al 9.
    pred = model.predict(roi_vec, verbose=0)
    numero    = int(np.argmax(pred)) # Argmax elige el índice (dígito) con el porcentaje más alto.
    confianza = float(np.max(pred))  # Guarda ese porcentaje de victoria.

    print(f"  [Debug] ROI procesada en X:{x} Y:{y} -> Predice: {numero} al {confianza*100:.1f}%")

    # Si la predicción es mediocre, la marcamos de ROJO como "Descartada" para no engañar al usuario.
    if confianza < CONFIANZA_MINIMA:
        print(f"    -> Descartado por baja confianza (menor a {CONFIANZA_MINIMA*100}%)")
        cv2.rectangle(original,(x,y),(x+w,y+h),(0,0,255),2) # Bounding box rojo
        cv2.putText(original,f"{numero}? ({confianza*100:.1f}%)",
                    (x,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
        continue

    # Si todo salió bien, guardamos el número y lo encerramos en VERDE.
    numeros_detectados.append((x, numero, confianza))

    cv2.rectangle(original,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(original,f"{numero} ({confianza*100:.1f}%)",
                (x,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

# =========================================================
# 10. RESULTADO FINAL (SALIDA DE CONSOLA E IMÁGENES)
# =========================================================
print("\n" + "="*40)
if numeros_detectados:
    # Ordenamos de izquierda a derecha la lectura
    numeros_detectados.sort(key=lambda item: item[0])
    secuencia = ''.join(str(num) for _,num,_ in numeros_detectados)
    print(f"Número(s) detectado(s): {secuencia}")
    print("\nDetalle por dígito:")
    for pos, (_, num, conf) in enumerate(numeros_detectados):
        print(f"  Posición {pos+1}: {num}  |  Confianza: {conf*100:.1f} %")
else:
    print("No se detectaron números con confianza suficiente.")
    print(f"(Umbral actual: {CONFIANZA_MINIMA*100:.0f} %)")
print("="*40)

# =========================================================
# RENDERIZADO VISUAL
# =========================================================
# imshow levanta las ventanas nativas con el buffer de píxeles original modificado y la máscara procesada.
cv2.imshow("Resultado Final — Detecciones", original)
cv2.imshow("Threshold (zona procesada)", thresh)
print("\nPresione cualquier tecla sobre la ventana para cerrar...")
cv2.waitKey(0) # Pausa la ejecución de Python hasta que el usuario cierre las ventanas gráficas.
cv2.destroyAllWindows()

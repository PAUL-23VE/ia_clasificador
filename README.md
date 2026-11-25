# 🔢 Clasificador de dígitos con red neuronal

Red neuronal **multiclase** que reconoce dígitos escritos a mano (0–9). Se entrena con el dataset **MNIST** y después detecta y clasifica los números de cualquier imagen que elijas.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)

## 🧠 Cómo funciona

1. **Entrenamiento:** red densa de 784 → 128 → 64 → 10 neuronas (ReLU y softmax), 12 épocas con el optimizador Adam.
2. **Capas personalizadas:** restricción que mantiene los pesos entre −1 y 1 (evita que crezcan sin control) y capas con un único sesgo escalar.
3. **Evaluación:** precisión sobre el conjunto de prueba y **matriz de confusión**.
4. **Uso con imágenes reales:** eliges una imagen; OpenCV la binariza, encuentra cada dígito por contornos, lo recorta, lo centra a 28×28 y la red lo clasifica.

El modelo entrenado se guarda en `modelo_mnist.keras` para no volver a entrenar en cada ejecución.

## 🚀 Cómo ejecutarlo

```bash
pip install -r requirements.txt
python main.py
```

Al terminar el entrenamiento se abre una ventana para seleccionar la imagen con números.

---

<sub>⚽ Parte del portafolio de **Paul Velastegui** · [paul-23ve.github.io](https://paul-23ve.github.io) · [GitHub](https://github.com/PAUL-23VE)</sub>

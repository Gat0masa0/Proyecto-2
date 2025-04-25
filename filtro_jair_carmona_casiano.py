import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import cv2 
import time

print(f"Versiones de librerías:")
print(f"- TensorFlow: {tf.__version__}")
print(f"- NumPy: {np.__version__}")
print(f"- OpenCV: {cv2.__version__}")

# --- Parámetros Globales ---
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1) 
EPOCHS = 5 
BATCH_SIZE = 128

# --- 1. Carga y Preprocesamiento de Datos ---
print("\nCargando dataset MNIST...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"Datos MNIST cargados.")
print(f"Forma original x_train: {x_train.shape}, y_train: {y_train.shape}")

# Función para preprocesar imágenes (normalizar y añadir canal)
def preprocess_images(images):
    images = images.astype('float32') # Convertir antes de normalizar/expandir
    # Normalizar píxeles al rango [0, 1]
    images = images / 255.0
    # Añadir dimensión de canal (CNN espera [batch, height, width, channels])
    images = np.expand_dims(images, -1)
    return images

# Preprocesar imágenes originales
x_train_orig_processed = preprocess_images(x_train.copy())
x_test_orig_processed = preprocess_images(x_test.copy())
print(f"Forma x_train original procesada: {x_train_orig_processed.shape}")

# --- 2. Aplicación del Filtro Laplaciano ---
print("\nAplicando filtro Laplaciano...")

def apply_laplacian_filter(images):
    """Aplica el filtro Laplaciano y normaliza su valor absoluto."""
    images_laplacian = []
    for img in images:
        # Aplicar Laplaciano (usar CV_64F para manejar posibles valores negativos)
        laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)

        # Tomar el valor absoluto para visualizar la "fuerza" de la respuesta
        laplacian_abs = np.abs(laplacian)

        # Normalizar el resultado absoluto al rango [0, 1] para la CNN
        # Usamos CV_32F como tipo de dato de salida para la red neuronal
        laplacian_normalized = cv2.normalize(laplacian_abs, None, alpha=0, beta=1,
                                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        images_laplacian.append(laplacian_normalized)

    return np.array(images_laplacian)

# Aplicar filtro a las imágenes originales (antes de normalizar/expandir dims)
x_train_laplacian = apply_laplacian_filter(x_train.copy())
x_test_laplacian = apply_laplacian_filter(x_test.copy())
print("Filtro Laplaciano aplicado.")

# Preprocesar imágenes filtradas con Laplaciano (SOLO añadir canal, ya están normalizadas a [0,1])
# Nota: La función preprocess_images re-normalizaría a /255, lo cual no queremos aquí.
#       Hacemos solo el expand_dims.
x_train_laplacian_processed = np.expand_dims(x_train_laplacian.astype('float32'), -1)
x_test_laplacian_processed = np.expand_dims(x_test_laplacian.astype('float32'), -1)
print(f"Forma x_train Laplaciano procesada: {x_train_laplacian_processed.shape}")


# --- Visualización Opcional (Primeras imágenes Original vs Laplaciano) ---
n_display = 5
plt.figure(figsize=(10, 4))
plt.suptitle("Comparación: Original vs. Filtro Laplaciano (MNIST)")
for i in range(n_display):
    # Original (antes de procesar para CNN)
    ax = plt.subplot(2, n_display, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Orig {i+1}")
    plt.axis("off")

    # Laplaciano (ya normalizado a [0,1] por la función apply_laplacian_filter)
    ax = plt.subplot(2, n_display, i + 1 + n_display)
    # Usamos [:,:,0] para quitar la dimensión de canal si se añadió antes por error
    # o directamente usamos la variable antes de expand_dims
    plt.imshow(x_train_laplacian[i], cmap='gray')
    plt.title(f"Laplaciano {i+1}")
    plt.axis("off")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# --- 3. Preparar Etiquetas ---
# Convertir vectores de clase a matrices de clase binaria (one-hot encoding)
y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_test_cat = to_categorical(y_test, NUM_CLASSES)
print(f"\nForma y_train (one-hot): {y_train_cat.shape}")

# --- 4. Definir el Modelo CNN ---
def build_simple_cnn(input_shape, num_classes):
    """Construye una CNN simple con 2 capas Conv y 2 Dense."""
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        # Dropout(0.5), # Podría añadirse para regularización
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

# --- 5. Entrenar y Evaluar Modelo con Imágenes ORIGINALES ---
print("\n--- Entrenamiento con Imágenes Originales ---")
model_orig = build_simple_cnn(INPUT_SHAPE, NUM_CLASSES)
model_orig.summary()

start_time = time.time()
history_orig = model_orig.fit(x_train_orig_processed, y_train_cat,
                              batch_size=BATCH_SIZE,
                              epochs=EPOCHS,
                              verbose=1,
                              validation_split=0.1) # Usar 10% para validación
end_time = time.time()
print(f"Tiempo de entrenamiento (Originales): {end_time - start_time:.2f} segundos")

# Evaluar en el conjunto de test
score_orig = model_orig.evaluate(x_test_orig_processed, y_test_cat, verbose=0)
print(f"\nResultados con Imágenes Originales:")
print(f"Pérdida en Test (Original): {score_orig[0]:.4f}")
print(f"Precisión en Test (Original): {score_orig[1]:.4f}")


# --- 6. Entrenar y Evaluar Modelo con Imágenes LAPLACIANAS ---
print("\n--- Entrenamiento con Imágenes Filtradas (Laplaciano) ---")
# Volvemos a construir el modelo para empezar desde cero
model_laplacian = build_simple_cnn(INPUT_SHAPE, NUM_CLASSES)

start_time = time.time()
history_laplacian = model_laplacian.fit(x_train_laplacian_processed, y_train_cat,
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                verbose=1,
                                validation_split=0.1)
end_time = time.time()
print(f"Tiempo de entrenamiento (Laplaciano): {end_time - start_time:.2f} segundos")

# Evaluar en el conjunto de test filtrado
score_laplacian = model_laplacian.evaluate(x_test_laplacian_processed, y_test_cat, verbose=0)
print(f"\nResultados con Imágenes Filtradas (Laplaciano):")
print(f"Pérdida en Test (Laplaciano): {score_laplacian[0]:.4f}")
print(f"Precisión en Test (Laplaciano): {score_laplacian[1]:.4f}")

# --- 7. Comparación Final ---
print("\n--- Comparación de Precisión Final ---")
print(f"Precisión en Test (Imágenes Originales): {score_orig[1]:.4f}")
print(f"Precisión en Test (Imágenes con Laplaciano): {score_laplacian[1]:.4f}")

if score_laplacian[1] > score_orig[1]:
    print("\nConclusión: El filtro Laplaciano MEJORÓ la precisión en este caso.")
elif score_laplacian[1] < score_orig[1]:
    print("\nConclusión: El filtro Laplaciano EMPEORÓ la precisión en este caso.")
else:
    print("\nConclusión: El filtro Laplaciano obtuvo la MISMA precisión en este caso.")


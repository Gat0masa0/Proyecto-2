# Proyecto-2
Clasificación de imágenes MNIST con y sin filtro Laplaciano

Este proyecto compara el rendimiento de un modelo de clasificación (CNN)
utilizando la base de datos MNIST en dos versiones:
1. Imágenes originales.
2. Imágenes procesadas con el filtro Laplaciano para detección de bordes.

Objetivo: Evaluar si el preprocesamiento con el Laplaciano mejora el rendimiento
de la clasificación de dígitos manuscritos al resaltar bordes finos.

 --- Justificación del Filtro Laplaciano ---
 
 Filtro Utilizado: Operador Laplaciano (cv2.Laplacian)

 ¿Por qué se eligió?:
 1. Detección de Bordes Finos: El Laplaciano es un operador de segunda
    derivada, lo que lo hace muy sensible a cambios rápidos de intensidad
    y eficaz para detectar bordes finos y detalles en la imagen. Los bordes
    son características fundamentales de los dígitos manuscritos.
 2. Resalte de Detalles: A diferencia de los filtros de primera derivada
    (como Sobel), el Laplaciano puede resaltar puntos y líneas finas de
    manera más isotrópica (independiente de la dirección).
 3. Sensibilidad al Ruido (Punto de Interés): Es conocido por ser sensible
    al ruido. Evaluar su efecto en la clasificación es interesante: ¿ayudará
    resaltando bordes o perjudicará amplificando ruido? Este experimento
    ayudará a responderlo para MNIST con esta CNN.
 4. Alternativa a Sobel: Proporciona una representación de bordes diferente
    a la de Sobel (basada en la segunda derivada y cruces por cero),
    ofreciendo un punto de comparación sobre qué tipo de representación de
    bordes podría ser más útil para la CNN.

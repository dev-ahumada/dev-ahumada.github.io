---
layout: post
title: "Contar objetos en imagen con Python"
subtitle: "Enfoque simple para contar objetos en una imagen usando Python y cvlib"
date: 2021-10-02 00:00:00 -0400
background: "/img/posts/2021-05-15-contar-objetos-imagenes-python/2021-05-15-contar-objetos-imagenes-python-1.png"
categories: posts/machine-learning
tags: [python, opencv, cvlib]
---
Contar objetos en una imagen es una tarea de visión por computadora. Hay un montón de librerías de visión por computadora en Python que podemos usar para esta tarea. Está vez usaremos un enfoque muy simple para contar objetos en una imagen usando Python.

# ¿Cómo contar objetos en una imagen usando Python?
Hay muchas librerías de visión por computadora que podemos usar para esta tarea, como OpenCV, TensorFlow, Pytorch, Scikit-image y cvlib. Esta última es una librería de visión por computadora muy simple, de alto nivel y fácil de usar en Python.
Usando las características de esta librería, podemos contar la cantidad de objetos en una imagen usando Python. Para poder usar esta librería, debemos estár seguros de tener OpenCV y TensorFlow instalados en nuestro sistema. Se pueden instalar fácilmente usando el comando: `pip install cvlib`.  
  
# Contar objetos en una imagen usando Python
Ahora veremos cómo usar la librería `cvlib` para contar el número de objetos en una imagen usando Python. Primero leeremos una imagen usando la librería OpenCV, luego detectaremos todos los objetos particulares usando `cvlib`. La imagen que usaremos para esta tarea se muestra a continuación y la podemos descargar [**aquí**](https://drive.google.com/file/d/14hgyVrVxQ3vPNRO2GGHdUxxOpAgoug6n/view?usp=sharing)
![Vehículos](/img/posts/2021-05-15-contar-objetos-imagenes-python/vehiculos.jpg)

Como podemos ver, la imagen que estamos usando aquí para la tarea de contar objetos en una imagen usando Python contiene vehículos. Primero, detectaremos todos los vehículos en la imagen y después contaremos el número de autos que hay entre ellos. A continuación se muestra cómo podemos contar la cantidad de autos en una imagen usando Python:
```python
in [1]:     import cv2
            import numpy as np
            import matplotlib.pyplot as plt
            import cvlib as cv
            from cvlib.object_detection import draw_bbox
            from numpy.lib.polynomial import poly

            imagen = cv2.imread("vehiculos.jpg")
            cuadro, etiqueta, contar = cv.detect_common_objects(imagen)
            output = draw_bbox(imagen, cuadro, etiqueta, contar)
            plt.imshow(output)
            plt.show()
            print("El número de autos en la imagen es " + str(etiqueta.count('car')))
```
Después de ejecutar el código anterior, veremos una imagen como la que se muestra a continuación:
![Detección de objetos en imagen](/img/posts/2021-05-15-contar-objetos-imagenes-python/Figure_1.png)
Al cerrar nuestra imagen, veremos lo siguiente en nuestra terminal:
```
out [1]:    El número de autos en la imagen es 10
```
Es así como podemos contar el número de objetos utilizando la librería `cvlib` en Python. Recuerda que podemos utilizar esta librería para diversas tareas de visión por computadora.

Puedes descargar el código de este projecto [**aquí**](https://drive.google.com/file/d/1XbnAUrXnXEWMfxvkOcGeFCmf_l6pG6bt/view?usp=sharing)
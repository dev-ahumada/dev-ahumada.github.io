---
layout: post
title: "Caricaturizar una imagen"
subtitle: "Creación de una foto caricaturizada usando Python y OpenCV."
date: 2021-03-05 00:00:00 -0400
background: '/img/posts/2021-05-03-caricaturizar-imagen-opencv-python/2021-05-03-caricaturizar-imagen-opencv-python-1.png'
categories: posts/python
tags: [OpenCV, Tkinter]
---

# Qué es OpenCV?
Python tiene múltiples librerías para aplicaciones de la vida real. Una de esas librerías es OpenCV, la cual es usada en Visión Computacional. Dicha librería incluye aplicaciones como captura y procesamiento de videos e imágenes, siendo mayormente utilizada en transformación de imágenes, detección de objetos y reconocimiento facial.

# Qué vamos a desarrollar?
Al final de este artículo, nuestra meta es transformar una imagen en su caricatura. Para ello, construiremos una aplicación en Python que convertirá nuestra imagen en una caricatura usando OpenCV.

# Imagen original.
![Imagen Original](/img/posts/2021-05-03-caricaturizar-imagen-opencv-python/2021-05-03-caricaturizar-imagen-opencv-python-2.png)
---
# Pasos para desarrollar nuestro Caricaturizador de imágenes.
---
## Paso 1: Importar los módulos que vamos a necesitar.
Importaremos los siguientes módulos:  
- **cv2**: Es una librería altamente optimizada que se enfoca en aplicaciones en tiempo real.  

- **easygui**: Módulo para la programación GUI fácil y rápidamente en Python. 

- **Numpy**: Es una librería para Python con soporte para matrices y arrays grandes y multidimensionales, junto con una gran colección de funciones matemáticas de alto nivel para operar en estas matrices.  

- **Imageio**: Es una librería que proporciona una interfaz fácil para leer y escribir una amplia gama de datos de imágenes.

- **Matplotlib**: Librería para crear visualizaciones estáticas, animadas e interactivas en Python. 

- **os**: Librería que nos proporciona funciones para interactuar con el sistema operativo.
  

```python
import cv2
import easygui 
import numpy as np
import imageio
import sys
import matplotlib as mpl
import os
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
```
---

## Paso 2: Construir un administrador de archivos para seleccionar un archivo en particular
En este paso, construimos la ventana principal donde se encontrarán los botones, etiquetas e imágenes de nuestra aplicación.

```python
def cargar():
    rutaImagen = easygui.fileopenbox()
    caricaturizar(rutaImagen)
```
El código anterior lanza una ventana emergente para seleccionar un archivo de nuestro dispositivo cada vez que lo ejecutamos. El método `fileopenbox()` en el módulo `easyGUI` nos devuelve la ruta del archivo seleccionado en formato string.  

---

## Paso 3: ¿Cómo almacenamos una imagen?
Ahora debes preguntarte, ¿cómo nuestro programa procesará una imagen?. Debemos recordar que para nuestra computadora todo esta formado por números. Por lo tanto, en el siguiente código, usaremos Numpy para convertir nuestra imagen en un array.  
```python
def caricaturizar(ruta_imagen)
# leer la imagen
    imagenOriginal = cv2.imread(rutaImagen)
    imagenOriginal = cv2.cvtColor(imagenOriginal, cv2.COLOR_BGR2RGB)
# confirmar que la imagen ha sido seleccionada
    if imagenOriginal is None:
        print("No se encontró ninguna imagen. Elige un archivo apropiado")
        sys.exit()
    
    redimen1 = cv2.resize(imagen_original, (960, 540))
```
`imread()` es un método de `cv2` que es usado para almacenar imágenes en forma de arrays. Esto nos permite realizar operaciones de acuerdo con nuestras necesidades. La imagen es procesada como un array, cuyos valores representan los valores R, G, y B de cada pixel.

**NOTA**: redimensionamos la imagen después de cada transformación para desplegar todas las imágenes en una escala similar.

Para convertir una imagen en una caricatura se requiere de muchas transformaciones. Primero, la imagen es convertida a escala de grises. Después, la imagen en escala de grises es suavizada, y tratamos de extraer los bordes de la imagen. Finalmente, formamos una imagen a color y la unimos con los bordes. Esto crea una caricatura con los bordes y los colores de la imagen original realzados.

---

## Paso 4: Transformar imagen a escala de grises
```python
#Convertir imagen a escala de grises
    imagenEscalaGrises = cv2.cvtColor(imagenOriginal, cv2.COLOR_BGR2GRAY)
    redimen2 = cv2.resize(imagenEscalaGrises, (960, 540))
```
`cvtColor()` es un método de `cv2` que recibe una imagen como primer argumento y una [*conversión de espacio de colores*](https://docs.opencv.org/4.5.3/d8/d01/group__imgproc__color__conversions.html) como segundo argumento. En este caso hemos usado `BGR2GRAY` para convertir nuestra imagen a una escala de grises.    

El código anterior producirá el siguiente resultado:  
![Imagen en escala de grises](/img/posts/2021-05-03-caricaturizar-imagen-opencv-python/2021-05-03-caricaturizar-imagen-opencv-python-3.png)

---

## Paso 5: Suavizar imagen en escala de grises
```python
#Aplicar medianBlur para suavizar imagen
    escalaGriseSuavizada = cv2.meadianBlur(imagenEscalaGrises, 5)
    redimen3 = cv2.resize(escalaGriseSuavizada, (960, 540))
```
Para suavizar una imagen simplemente aplicamos un efecto de desenfoque. Esto lo hacemos usando la función `medianBlur()`, la cual reemplaza el elemento central de la imagen por la mediana de todos los pixeles en el área del kernel. Esta operación procesa los bordes mientras elimina el ruido, creando un efecto de desenfoque.    

El código anterior producirá el siguiente resultado:
![Imagen suavizada](/img/posts/2021-05-03-caricaturizar-imagen-opencv-python/2021-05-03-caricaturizar-imagen-opencv-python-4.png)

---

## Paso 6: Recuperar los bordes de la imagen
```python
#recuperando los bordes de imagen para efecto caricatura
    bordesImagen = cv2.adaptiveThreshold(escalaGriseSuavizada, 255, 
                    cv2.ADAPTIVE_THRESH_MEAN_C, 
                    cv2.THRESH_BINARY, 9, 9)

    redimen4 = cv2.resize(bordesImagen, (960,540))
```
El efecto de caricatura tiene dos características:
1. Bordes remarcados
2. Colores suaves  

En este paso, trabajaremos en la primer característica. Trataremos de recuperar los bordes y remarcarlos. Esto lo logramos con el uso del método `adaptiveThreshold()`.    

El código anterior producirá el siguiente resultado:
![Bordes](/img/posts/2021-05-03-caricaturizar-imagen-opencv-python/2021-05-03-caricaturizar-imagen-opencv-python-5.png)

---

## Paso 7: Preparar imagen a color
```python
#aplicar filtro bilateral para remover ruido
    imagenColor = cv2.bilateralFilter(imagenOriginal, 9, 300, 300)
    redimen5 = cv2.resize(imagenColor, (960, 540))
```
En el código anterior, trabajamos en la segunda característica. Hemos preparado una imagen colorida para unirla con los bordes y crear un efecto de caricatura. Usando el método `bilateralFilter()` removemos el ruido y la aspereza en los colores.    

El código anterior producirá el siguiente resultado:  
![Imagen a color](/img/posts/2021-05-03-caricaturizar-imagen-opencv-python/2021-05-03-caricaturizar-imagen-opencv-python-6.png)

---
## Paso 8: Aplicar efecto de caricatura a imagen
```python
#unir bordes con nuestra imagen a color
    imagenCaricaturizada = cv2.bitwise_and(imagenColor, imagenColor, mask=bordesImagen)
    redimen6 = cv2.resize(imagenCaricaturizada, (960, 540))
```
Es momento de combinar las dos características. Esto podemos lograrlo realizando un "enmascarado". Mediante el método `bitwise_and` enmascaramos las dos imágenes previamente obtenidas. Esto finalmente "caricaturizará" nuestra imagen.   

El código anterior producirá el siguiente resultado: 
![Imagen caricaturizada](/img/posts/2021-05-03-caricaturizar-imagen-opencv-python/2021-05-03-caricaturizar-imagen-opencv-python-7.png)

---

## Paso 9: Plotear todas las transiciones juntas
```python
#Plotear todas las transiciones
    imagenes = [redimen1, redimen2, redimen3, redimen4, redimen5, redimen6]
    fig, axes = plt.subplots(3, 2, figsize=(8.8),
                subplot_kw={'xticks':[], 'yticks':[]}, 
                gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')

    //código botón guardado

    plt.show
```
Para plotear todas las imágenes, primero debemos hacer una lista de ellas. La lista es llamada `imagenes` y contiene todas las imágenes redimensionadas. Después, creamos una figura para plotear una a una las imágenes en cada subplot usando el método `imshow()`.  
Finalmente, la función `plt.show()` plotea todas las transiciones en un solo plot.    

El código anterior producirá el siguiente resultado: 
![Todas las imágenes](/img/posts/2021-05-03-caricaturizar-imagen-opencv-python/2021-05-03-caricaturizar-imagen-opencv-python-1.png)

---

## Paso 10: Crear la ventana principal
```python
ventanaPrincipal = tk.Tk()
ventanaPrincipal.geometry('400x400')
ventanaPrincipal.title('Caricaturiza tu imagen')
ventanaPrincipal.configure(background='white')
etiqueta = Label(ventanaPrincipal, background='#CDCDCD', font=('calibri', 20, 'bold'))
```
---

## Paso 11: Crear botón caricaturizar en ventana principal
```python
botonCaricaturizar = Button(ventanaPrincipal, text="Caricaturizar una imagen", command=cargar, padx=10, pady=5)
botonCaricaturizar.configure(background='#364156', foreground='white', font=('calibri', 10, 'bold'))
botonCaricaturizar.pack(side=TOP, pady=50)
```
![Boton caricaturizar](/img/posts/2021-05-03-caricaturizar-imagen-opencv-python/2021-05-03-caricaturizar-imagen-opencv-python-8.png)

---

## Paso 13: Crear botón "Guardar" en ventana principal
```python
botonGuardar = Button(ventanaPrincipal, text='Guardar imagen caricaturizada', command=lambda: guardar(redimen6, rutaImagen), padx=30, pady=5)
botonGuardar.configure(background='#364156', foreground='white', font=('calibri', 10, 'bold'))
botonGuardar.pack(side=TOP, pady=50)
```
El código anterior crea un botón tan pronto como la transformación de la imagen es hecha. Dándole al usuario la opción de guardar la imagen caricaturizada.

![Boton guardado](/img/posts/2021-05-03-caricaturizar-imagen-opencv-python/2021-05-03-caricaturizar-imagen-opencv-python-9.png)

---

## Paso 10: Crear mensaje de guardado
```python
def guardar(redimen6, rutaImagen):
    #guardar imagen usando imwrite()
    nombreNuevo = "imagen_caricaturizada"
    ruta1 = os.path.dirname(rutaImagen)
    extension = os.path.splitext(rutaImagen)[1]
    ruta = os.path.join(ruta1, nombreNuevo+extension)
    cv2.imwrite(ruta, cv2.cvtColor(redimen6, cv2.COLOR_RGB2BGR))
    mensaje = "Imagen guardada como " + nombreNuevo + "en " + path
    messagebox.showinfo(title=None, message=mensaje)
```
En este paso, la idea es guardar nuestra imagen caricaturizada. Para esto, tomamos `rutaImagen` y cambiamos el nombre del antiguo archivo por un nuevo nombre. Posteriormente, almacenamos la imagen caricaturizada en la misma carpeta que `imagenOriginal` agregando el nuevo nombre al título del archivo.  

Para esto, extraemos el título del archivo usando el método `os.path.dirname()`. Además, el método `os.path.splitext()` es usado para extraer la extensión del archivo.    

La variable `nombreNuevo` almacena `"imagen_caricaturizada"` como el nombre de un nuevo archivo. Mientras que la expresión `os.path.join(ruta1, nombreNuevo+extension)` concatena el directorio del archivo con su nuevo nombre y extensión. Esto forma la ruta completa para nuestro nuevo archivo.

El método `imwrite()` de la librería `cv2` es usado para guardar el archivo en la ruta mencionada anteriormente. La expresión `cv2.cvtColor(redimen6, cv2.COLOR_RGB2BGR)` es usada para asegurar que ningún color se extraiga o resalte mientras guardamos nuestra imagen. Así, finalmente, se le da al usuario la confirmación de que la imagen ha sido guardada con el nombre y la ruta del archivo.

![Mensaje guardado](/img/posts/2021-05-03-caricaturizar-imagen-opencv-python/2021-05-03-caricaturizar-imagen-opencv-python-10.png)

---

## Paso 14: Función principal para ejecutar Tkinter
```python
ventanaPrincipal.mainloop()
```
---
# Resultado final
![Resultado final](/img/posts/2021-05-03-caricaturizar-imagen-opencv-python/2021-05-03-caricaturizar-imagen-opencv-python-11.png)

---
# Descargar
Puedes descargar el código fuente del projecto *Caricaturizar imagen en Python* [aquí](https://drive.google.com/file/d/1Ius1RFI-0CzkL89fOmb-EZfBtKrUp4lB/view?usp=sharing)
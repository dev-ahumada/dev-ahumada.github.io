---
layout: post
title: "Predecir el valor de mercado con aprendizaje automático"
subtitle: "Predicción del precio de las acciones de Microsoft con aprendizaje automático"
date: 2021-10-11 00:00:00 -0400
background: "/img/posts/2021-05-17-predecir-valor-mercado-machine-learning/2021-05-17-predecir-valor-mercado-machine-learning-1.jpeg"
categories: posts/machine-learning
tags: [python, numpy, pandas, seaborn]
---
Microsoft es hoy una de las empresas de tecnología más grandes con más de 160,000 empleados en todo el mundo. Es conocido por el sistema operativo Windows, que es uno de los sistemas operativos informaticos más populares. En este artículo, vamos a predecir los futuros precios de las acciones de Microsoft con aprendizaje automático usando Python.
# Predicción del precio de las acciones de Microsoft
Windows 10 es uno de los sistemas operativos más usados en el mundo. Cuando Microsoft lanzó Windows 10, se anunció que sería la última versión de Windows y que después de eso Microsoft solo trabajaría en sus actualizaciones. Pero ahora, una vez más Microsoft ha atraído la atención del mundo con el lanzamiento de Windows 11. Así que sería un buen momento para predecir el precio de las acciones de Microsoft.
# Predicción del precio de las acciones de Microsoft usando Python
Para comenzar con la tarea de pronosticar los precios de las acciones de Microsoft, primero debemos tener un conjunto de datos. Entonces, simplemente debemos seguir los pasos mencionados a continuación:
1.  Ir al sitio web de [Yahoo Finanzas](https://es-us.finanzas.yahoo.com/)
2.  Buscar **"MSFT"**
3.  Hacer click en **"Datos históricos"**
4. Hacer click en **"Descargar"**  

Después de completar estos pasos, tendremos un conjunto de datos de los precios históricos de las acciones de Microsoft en nuestra carpeta de descargas. Ahora, podemos iniciar con la tarea de predecir los precios de las acciones de Microsoft importando las librerías de Ptyhon necesarias y el conjunto de datos:  
```python
in [1]:     import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set()
            plt.style.use('fivethirtyeight')

            data = pd.read_csv("MSFT.csv")
            print(data.head())
```
```
out [1]        Date        Open        High        Low         Close       Adj Close   Volume 
            0  2020-05-18  185.750000  186.199997  183.960007  184.910004  182.279190  35264500 
            1  2020-05-19  185.029999  186.600006  183.490005  183.630005  181.017380  26799100 
            2  2020-05-20  184.809998  185.850006  183.940002  185.660004  183.528229  31261300 
            3  2020-05-21  185.399994  186.669998  183.289993  183.429993  181.323822  29119500  
            4  2020-05-22  183.190002  184.460007  182.539993  183.509995  181.402908  20826900   
```
En este conjunto de datos, la columna `Close` contiene los valores que queremos predecir en el futuro. Así que echemos un vistazo de más de cerca a los precios de cierre históricos de las acciones de Microsoft:
```python
in [2]:     plt.figure(figsize=(10, 4))
            plt.title("Precio Acciones Microsoft")
            plt.xlabel("Fecha")
            plt.ylabel("Cierre")
            plt.plot(data["Close"])
            plt.show()
```
![Precio acciones Microsoft](/img/posts/2021-05-17-predecir-valor-mercado-machine-learning/2021-05-17-predecir-valor-mercado-machine-learning-1.png)

Ahora, revisamos la correlación entre las columnas de nuestro conjunto de datos:
```python
in [3]:     print(data.corr())
            sns.heatmap(data.corr())
            plt.show()
```
```
out [3]:               Open      High      Low       Close      Adj Close   Volume
            Open       1.000000  0.995421  0.994285  0.988295   0.988418   -0.194574
            High       0.995421  1.000000  0.994375  0.994169   0.994168   -0.169893
            Low        0.994285  0.994375  1.000000  0.995694   0.995829   -0.237993
            Close      0.988295  0.994169  0.995694  1.000000   0.999857   -0.215556
            Adj Close  0.988418  0.994168  0.995829  0.999857   1.000000   -0.216663
            Volume    -0.194574 -0.169893 -0.237993 -0.215556  -0.216663    1.000000
```
![Correlación conjunto de datos Microsoft](/img/posts/2021-05-17-predecir-valor-mercado-machine-learning/2021-05-17-predecir-valor-mercado-machine-learning-2.png)

La correlación explica cómo se relacionan una o más variables entre sí.
-   **Correlación positiva:** dos columnas (variables) pueden correlacionarse positivamente entre sí. Esto significa que cuando el valor de una variable aumenta, el valor de las otras variables también aumenta y viceversa.  

-   **Correlación negativa:** dos columnas (variables) pueden correlacionarse negativamente entre sí. Esto significa que cuando el valor de una variable aumenta, el valor de las otras variables disminuye y viceversa.

-   **Sin correlación:** dos columnas (variables) pueden no correlacionarse entre sí. Esto significa que cuando el valor de una variable aumenta o disminuye, el valor de las otras variables no cambia.

Ahora preparamos los datos para que se ajusten al modelo de aprendizaje automático. En este proceso, primero agregamos las variables más importantes a `x` y la columna objetivo en `y`. Posteriormente, dividimos el conjunto de datos en conjuntos de entrenamientos y de prueba:
```python
in [4]:     X = data[["Open", "High", "Low"]]
            y = data["Close"]
            X = X.to_numpy()
            y = y.to_numpy()
            y = y.reshape(-1, 1)

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
A continuación, usamos el algoritmo de árboles de decisión para el problema de  regresión y entrenamos el modelo de predicción del precio de las acciones de Microsoft. Después revisaramos el precio pronosticado de las acciones para los próximos 5 días:
```python
in [5]:     from sklearn.tree import DecisionTreeRegressor
            
            modelo = DecisionTreeRegressor()
            modelo.fit(X_train, y_train)
            ypron = modelo.predict(X_test)
            data = pd.DataFrame(data={"Precio pronosticado": ypron})
            print(data.head())
```
```
out [5]:        Precio pronosticado
            0   224.970001
            1   181.399994
            2   219.619995
            3   211.600006
            4   213.289993
```
En resumen, así es como podemos predecir los precios de las acciones de Microsoft con Aprendizaje Automático usando Python.

Puedes descargar el cuaderno del projecto [aquí](https://drive.google.com/file/d/1vBkkUKNHFuQOSkvimc5Nw8ZIRhSUX9T8/view?usp=sharing)
---
layout: post
title: "Análisis de Google Search usando Python"
subtitle: ""
date: 2021-10-25 00:00:00 -0400
background: "/img/posts/2021-05-20-analisis-google-search-python/2021-05-20-analisis-google-search-python-bckgrd.jpeg"
categories: posts/machine-learning
tags: [python, pandas, pytrends, matplotlib]
---
Aproximadamente 3.5 billones de búsquedas se realizan en Google Search diariamente, lo que significa que alrededor de 40,000 búsquedas se realizan cada segundo. Por lo tanto, Google Search es un gran caso de uso para analizar datos basados en consultas de búsqueda. Con eso en mente, analizaremos Google Search con el uso de Python.  

# Análisis de Google Search con Python
Google no da mucho acceso a los datos sobre las consultas de búsqueda diarias, pero otra aplicación de Google conocida como Google Trends se puede utilizar para la tarea de análisis de búsqueda de Google. Google Trends proporciona una API que se puede utilizar para analizar las búsquedas diarias en Google. Esta API se conoce como `pytrends`, podemos instalarla fácilmente usando el comando `pip install pytrends`.

Una vez instalada la librería `pytrends` en nuestro sistema, comenzamos con la tarea del análisis de Google Search importando las librerías de Python que vamos a necesitar:
```python
in [1]:     import pandas as pd
            from pytrends.request import TrendReq
            import matplotlib.pyplot as plt
            tendencias = TrendReq()
```
Aquí analizamos las tendencias de búsqueda de Google en las consultas basadas en "Machine Learning", creamos un `DataFrame` de los 10 principales países que buscan "Machine Learning" en Google:
```python
in [2]:     tendencias.build_payload(kw_list=["Machine Learning"])
            data = tendencias.interest_by_region()
            data = data.sort_values(by="Machine Learning", ascending=False)
            data = data.head(10)
            print(data)
```
```
out [2]:                 Machine Learning
            geoName                      
            China                     100
            Singapore                  39
            St. Helena                 28
            India                      27
            Hong Kong                  21
            South Korea                19
            Nepal                      18
            Pakistan                   17
            Israel                     15
            Bangladesh                 15
```
De acuerdo con los resultados anteriores, las consultas basadas en "Machine Learning" se realizan principalmente en China. También podemos visualizar estos datos usando un gráfico de barras:
```python
in [3]:     data.reset_index().plot(x="geoName", y="Machine Learning", 
                        figsize=(20,15), kind="bar")
            plt.style.use('fivethirtyeight')
            plt.xlabel('País')
            plt.ylabel('Machine Learning')
            plt.show()
```
![Gráfico de barras](/img/posts/2021-05-20-analisis-google-search-python/2021-05-20-analisis-google-search-python-1.png)  

Como sabemos, el Machine Learning ha sido el foco de muchas empresas y estudiantes durante los últimos 3-4 años. Veamos la tendencia de las consultas en Google Search basadas en "Machine Learning" para comprender cómo incrementa o disminuye.
```python
in [4]:     data = TrendReq(hl='en-US', tz=360)
            data.build_payload(kw_list=['Machine Learning'])
            data = data.interest_over_time()
            fig, ax = plt.subplots(figsize=(20, 15))
            data['Machine Learning'].plot()
            plt.style.use('fivethirtyeight')
            plt.title('Total búsquedas Google Search para Machine Learning', fontweight='bold')
            plt.xlabel('Año')
            plt.ylabel('Total Búsquedas')
            plt.show()
```
![Gráfico de barras](/img/posts/2021-05-20-analisis-google-search-python/2021-05-20-analisis-google-search-python-2.png)  

Podemos ver como las búsquedas de Google basadas en "Machine Learning" comenzaron a aumentar en 2017 y llegaron a su punto máximo en 2020.  

Así es como podemos analizar las búsquedas de Google en función de cualquier palabra clave y comprender qué buscan las personas en Google en un momento dado.  

Puedes descargar el cuaderno del este projecto [aquí](https://drive.google.com/file/d/1iy1mQ6z_xDOxjo5-73hn_NA-mF8oyapl/view?usp=sharing)
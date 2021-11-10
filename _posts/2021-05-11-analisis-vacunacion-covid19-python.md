---
layout: post
title: "Análisis de vacunas Covid-19 con Python"
subtitle: "Cuantas y cuales vacunas está usando cada país para combatir el COVID-19."
date: 2021-05-11 00:00:00 -0400
background: "/img/posts/2021-05-11-analisis-vacunacion-covid19-python/2021-05-11-analisis-vacunacion-covid19-python-1.png"
categories: posts/machine-learning
tags: [python, pandas, numpy, plotly]
---
# Análisis de las vacunas de COVID-19 con Python

El [conjunto de datos](https://drive.google.com/file/d/1j6ph-K_vFo5BSKoXr4QuYaBaOnBZKm4k/view?usp=sharing) que usaremos para el análisis de las vacunas de COVID-19 fue tomada de Kaggle. Comenzaremos por importar las librerías de Python necesarias y el conjunto de datos:

```python
in [1]:     import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            data = pd.read_csv("vacunacion_paises.csv")
            data.head()
```

```
out [1]:     	country      iso_code  date 	   total_vaccinations 	people_vaccinated  people_fully_vaccinated  daily_vaccinations_raw  
            0 	Afghanistan  AFG       2021-02-22  0.0 	                0.0 	           NaN 	                    NaN
            1 	Afghanistan  AFG       2021-02-23  NaN 	                NaN 	           NaN 	                    NaN
            2 	Afghanistan  AFG       2021-02-24  NaN 	                NaN 	           NaN 	                    NaN
            3 	Afghanistan  AFG       2021-02-25  NaN 	                NaN 	           NaN 	                    NaN
            4 	Afghanistan  AFG       2021-02-26  NaN 	                NaN 	           NaN 	                    NaN
```  
---  

>**NOTA:** El conjunto de datos tiene más columnas de las que se muestran aquí.  

---  

A continuación, exploraremos un poco más el conjunto de datos. La opción `max_rows` nos permite controlar el número de columnas que `pandas` imprime. Recibe un entero o  `None` para imprimir todas las filas.
```python
in [2]:     pd.set_option("max_rows", None)
```
Después, contamos el número de entradas que tiene cada país en nuestro conjunto de datos.
```python
in [3]:     data.country.value_counts()
```
```
out [3]:    Scotland                290
            England                 290
            Seychelles              290
            Serbia                  290
            Gibraltar               289
            United Kingdom          289
            Wales                   289
            Northern Ireland        289
            Name: country, Length: 175, dtype: int64
```
El Reino Unido está conformado por Inglaterra, Escocia, Gales e Irlanda del Norte. Pero en los datos anteriores, estos países se mencionan por separado con valores muy parecidos. Esto puede ser un error al registrar los datos. Podemos arreglar esto de la siguiente manera:
```python
in [4]:     data = data[data.country.apply(lambda x: x not in ["Scotland", "England", "Wales", "Northern Ireland"])]
            data.country.value_counts()
```
```
out [4]:    Serbia                  290
            Seychelles              290
            United Kingdom          289
            Gibraltar               289
            Singapore               288
            Albania                 288
            Jordan                  288
            Indonesia               288
            Name: country, Length: 171, dtype: int64
```
Ahora, vamos a revisar las vacunas disponibles en este conjunto de datos:
```python
in [5]:     data.vaccines.value_counts()
```
```
out [5]:    Johnson&Johnson, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech                                            7102
            Oxford/AstraZeneca                                                                                       4867
            Oxford/AstraZeneca, Pfizer/BioNTech                                                                      2963
            Moderna, Oxford/AstraZeneca, Pfizer/BioNTech                                                             2910
            Moderna, Pfizer/BioNTech                                                                                 2194
            Oxford/AstraZeneca, Sinopharm/Beijing                                                                    2174
            Pfizer/BioNTech                                                                                          2001
            Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sputnik V                                        1876
            Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing                                                   1532
            Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac, Sputnik V                                                  1102
            Johnson&Johnson, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sputnik V                       1015
            Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinovac                                           966
            CanSino, Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac                                                     823
            Oxford/AstraZeneca, Sputnik V                                                                             728
            Oxford/AstraZeneca, Sinopharm/Beijing, Sinovac, Sputnik V                                                 728
            Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing                                           701
            Johnson&Johnson, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinovac, Sputnik V      660
            Johnson&Johnson, Moderna, Pfizer/BioNTech                                                                 640
            Johnson&Johnson, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinovac, Sputnik V               621
            Johnson&Johnson, Oxford/AstraZeneca                                                                       582
            Sinopharm/Beijing                                                                                         568
            Johnson&Johnson, Oxford/AstraZeneca, Sinopharm/Beijing                                                    562
            Pfizer/BioNTech, Sinovac                                                                                  534
            Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac                                                              524
            Johnson&Johnson, Pfizer/BioNTech                                                                          518
            Sinopharm/Beijing, Sputnik V                                                                              507
            Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sputnik V                                505
            Moderna                                                                                                   489
            Oxford/AstraZeneca, Pfizer/BioNTech, Sputnik V                                                            486
            Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac                                                     456
            Covaxin, Oxford/AstraZeneca, Sinopharm/Beijing                                                            452
            Johnson&Johnson, Oxford/AstraZeneca, Sinopharm/Beijing, Sinovac                                           443
            Johnson&Johnson, Oxford/AstraZeneca, Pfizer/BioNTech                                                      433
            Johnson&Johnson, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing                                   417
            Oxford/AstraZeneca, Sinovac                                                                               343
            CanSino, Sinopharm/Beijing, Sinopharm/Wuhan, Sinovac, ZF2001                                              316
            EpiVacCorona, Sputnik V                                                                                   313
            CanSino, Johnson&Johnson, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac, Sputnik V                305
            CanSino, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sputnik V                       302
            Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sputnik V                                                   296
            Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinopharm/Wuhan, Sputnik V                        295
            Johnson&Johnson, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sputnik V               294
            Oxford/AstraZeneca, Sinopharm/Beijing, Sputnik V                                                          290
            Moderna, Pfizer/BioNTech, Sinovac                                                                         288
            Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinovac                                  288
            Covaxin, Oxford/AstraZeneca, Sputnik V                                                                    285
            Johnson&Johnson, Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac                                             284
            QazVac, Sinopharm/Beijing, Sputnik V                                                                      269
            CanSino, Covaxin, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinovac, Sputnik V     267
            Covaxin, Johnson&Johnson, Oxford/AstraZeneca, Sinopharm/Beijing, Sputnik V                                267
            Pfizer/BioNTech, Sinopharm/Beijing                                                                        260
            COVIran Barekat, Covaxin, Oxford/AstraZeneca, Sinopharm/Beijing, Soberana02, Sputnik V                    260
            Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinovac, Sputnik V                                250
            Johnson&Johnson, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac                                    250
            Covaxin, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinovac, Sputnik V              243
            Moderna, Oxford/AstraZeneca                                                                               243
            Johnson&Johnson, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sputnik V                                  237
            Abdala, Sinopharm/Beijing, Sinopharm/Wuhan, Sputnik V                                                     234
            Johnson&Johnson, Oxford/AstraZeneca, Sinopharm/Beijing, Sinovac, Sputnik Light, Sputnik V                 226
            Pfizer/BioNTech, Sputnik V                                                                                226
            Johnson&Johnson, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinovac                          225
            Medigen, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech                                                     220
            Moderna, Oxford/AstraZeneca, Sputnik V, ZF2001                                                            210
            Sputnik V                                                                                                 210
            Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sputnik Light, Sputnik V                 195
            Moderna, Oxford/AstraZeneca, Sinopharm/Beijing, Sputnik V                                                 191
            EpiVacCorona, Oxford/AstraZeneca, Sinopharm/Beijing, Sputnik V                                            183
            Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac, Sputnik V                                          167
            Covaxin, Oxford/AstraZeneca                                                                               163
            Abdala, Soberana02                                                                                        159
            Johnson&Johnson, Oxford/AstraZeneca, Sinovac                                                              142
            Johnson&Johnson, Moderna                                                                                   95
            Johnson&Johnson                                                                                            67
            Name: vaccines, dtype: int64
```
Tenemos una gran variedad de vacunas COVID-19 disponibles en este conjunto de datos. Ahora, creamos un `DataFrame` nuevo donde seleccionamos solo la vacuna y los países para explorar qué vacuna se aplicó en cada país:
```python
in [6]:     df = data[['vaccines', 'country']]
            df.head()
```
```
out [6]:       vaccines 	                                        country
            0  Johnson&Johnson, Oxford/AstraZeneca, Pfizer/Bi... 	Afghanistan
            1  Johnson&Johnson, Oxford/AstraZeneca, Pfizer/Bi... 	Afghanistan
            2  Johnson&Johnson, Oxford/AstraZeneca, Pfizer/Bi... 	Afghanistan
            3  Johnson&Johnson, Oxford/AstraZeneca, Pfizer/Bi... 	Afghanistan
            4  Johnson&Johnson, Oxford/AstraZeneca, Pfizer/Bi... 	Afghanistan
```
A continuación, vamos a ver cuántos países están tomando cada una de las vacunas mencionadas en este conjunto de datos:
```python
in [7]:     dict_ = {}
            for i in df.vaccines.unique():
                dict_[i] = [df["country"][j] for j in df[df["vaccines"]==i].index]

            vaccines = {}
            for key, value in dict_.items():
                vaccines[key] = set(value)
            for i, j in vaccines.items():
                print(f"{i} -->> {j}\n\n")
```
```
out [7]:    Johnson&Johnson, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing -->> {'Cameroon', 'Afghanistan'}

            Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac, Sputnik V -->> {'Bosnia and Herzegovina', 'Oman', 'Azerbaijan', 'Albania'}

            Oxford/AstraZeneca, Sinopharm/Beijing, Sinovac, Sputnik V -->> {'Zimbabwe', 'Algeria', 'Armenia'}

            Oxford/AstraZeneca, Pfizer/BioNTech -->> {'Saint Lucia', 'Panama', 'Costa Rica', 'Saint Kitts and Nevis', 'Andorra', 'Bermuda', 'Saudi Arabia', 'Slovenia', 'Cayman Islands', 'Grenada', 'Kosovo'}

            Oxford/AstraZeneca -->> {'Papua New Guinea', 'Vanuatu', 'Mali', 'Falkland Islands', 'Anguilla', 'Sao Tome and Principe', 'Madagascar', 'Solomon Islands', 'Kiribati', 'Togo', 'Democratic Republic of Congo', 'Nauru', 'Lesotho', 'Fiji', 'Tonga', 'Liberia', 'Ethiopia', 'Saint Vincent and the Grenadines', 'Pitcairn', 'Nigeria', 'Saint Helena', 'Montserrat', 'Samoa', 'Uganda', 'Tuvalu', 'Niue', 'Angola'}

            Oxford/AstraZeneca, Pfizer/BioNTech, Sputnik V -->> {'Nicaragua', 'Antigua and Barbuda'}

            CanSino, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sputnik V -->> {'Argentina'}

            Pfizer/BioNTech -->> {'Kuwait', 'Cook Islands', 'New Caledonia', 'New Zealand', 'Monaco', 'Turks and Caicos Islands', 'Tokelau', 'Aruba', 'Gibraltar'}

            Moderna, Oxford/AstraZeneca, Pfizer/BioNTech -->> {'United Kingdom', 'Canada', 'Australia', 'Isle of Man', 'Sint Maarten (Dutch part)', 'Rwanda', 'Guernsey', 'Japan', 'Sweden', 'Jersey', 'Finland'}

            Johnson&Johnson, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech -->> {'Germany', 'Croatia', 'Portugal', 'Latvia', 'Ireland', 'Romania', 'South Korea', 'Poland', 'Cyprus', 'Czechia', 'Austria', 'Spain', 'France', 'Iceland', 'Jamaica', 'Estonia', 'Luxembourg', 'Italy', 'Belgium', 'Malta', 'Netherlands', 'Bulgaria', 'Greece', 'Lithuania'}

            Johnson&Johnson, Oxford/AstraZeneca, Pfizer/BioNTech -->> {'Bahamas', 'Eswatini'}

            Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sputnik V -->> {'Serbia', 'Jordan', 'Bahrain', 'Iraq', 'Mongolia', 'Montenegro', 'Lebanon'}

            Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing -->> {'Cape Verde', 'Bangladesh', 'Bhutan'}

            Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing -->> {'Maldives', 'Suriname', 'Peru', 'Trinidad and Tobago', 'Barbados', 'Dominica'}

            Sinopharm/Beijing, Sputnik V -->> {'Kyrgyzstan', 'Belarus'}

            Oxford/AstraZeneca, Sinopharm/Beijing -->> {'Brunei', 'Namibia', 'Niger', 'Mozambique', 'Guinea-Bissau', 'Myanmar', 'Mauritania', 'Senegal', 'Sierra Leone', 'Belize'}

            Oxford/AstraZeneca, Sinovac -->> {'Timor', 'Benin'}

            Johnson&Johnson, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sputnik V -->> {'Bolivia', 'Morocco', 'Moldova', "Cote d'Ivoire"}

            Moderna, Pfizer/BioNTech -->> {'Bonaire Sint Eustatius and Saba', 'Qatar', 'Liechtenstein', 'Switzerland', 'Curacao', 'Faeroe Islands', 'Israel', 'Norway'}

            Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac -->> {'Botswana', 'Ukraine'}

            Johnson&Johnson, Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac -->> {'Brazil'}

            Johnson&Johnson, Oxford/AstraZeneca -->> {'British Virgin Islands', 'Malawi', 'South Sudan'}

            Johnson&Johnson, Oxford/AstraZeneca, Sinopharm/Beijing -->> {'Burkina Faso', 'Zambia', 'Gambia'}

            Sinopharm/Beijing -->> {'Burundi', 'Equatorial Guinea', 'Gabon', 'Chad'}

            Johnson&Johnson, Oxford/AstraZeneca, Sinopharm/Beijing, Sinovac -->> {'Cambodia', 'Somalia'}

            Covaxin, Oxford/AstraZeneca -->> {'Central African Republic'}

            CanSino, Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac -->> {'Chile', 'Malaysia', 'Ecuador'}

            CanSino, Sinopharm/Beijing, Sinopharm/Wuhan, Sinovac, ZF2001 -->> {'China'}

            Johnson&Johnson, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac -->> {'Colombia'}

            Covaxin, Oxford/AstraZeneca, Sinopharm/Beijing -->> {'Comoros', 'Mauritius'}

            Moderna, Oxford/AstraZeneca, Sinopharm/Beijing, Sputnik V -->> {'Congo'}

            Abdala, Soberana02 -->> {'Cuba'}

            Johnson&Johnson, Moderna, Pfizer/BioNTech -->> {'Denmark', 'United States'}

            Johnson&Johnson, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinovac, Sputnik V -->> {'Egypt', 'Laos', 'Djibouti'}

            Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinovac -->> {'Dominican Republic', 'El Salvador', 'Georgia', 'Thailand'}

            Johnson&Johnson, Pfizer/BioNTech -->> {'South Africa', 'French Polynesia'}

            Oxford/AstraZeneca, Sputnik V -->> {'Kenya', 'Ghana', 'Guyana'}

            Moderna -->> {'Wallis and Futuna', 'Greenland'}

            Moderna, Oxford/AstraZeneca -->> {'Guatemala'}

            Sputnik V -->> {'Guinea'}

            Johnson&Johnson, Moderna -->> {'Haiti'}

            Johnson&Johnson, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sputnik V -->> {'Honduras'}

            Pfizer/BioNTech, Sinovac -->> {'Hong Kong', 'Turkey'}

            Johnson&Johnson, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sputnik V -->> {'Hungary'}

            Covaxin, Oxford/AstraZeneca, Sputnik V -->> {'India'}

            Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinovac -->> {'Indonesia'}

            COVIran Barekat, Covaxin, Oxford/AstraZeneca, Sinopharm/Beijing, Soberana02, Sputnik V -->> {'Iran'}

            QazVac, Sinopharm/Beijing, Sputnik V -->> {'Kazakhstan'}

            Johnson&Johnson, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinovac, Sputnik V -->> {'Tunisia', 'Philippines', 'Libya'}

            Pfizer/BioNTech, Sinopharm/Beijing -->> {'Macao'}

            CanSino, Johnson&Johnson, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac, Sputnik V -->> {'Mexico'}

            Covaxin, Johnson&Johnson, Oxford/AstraZeneca, Sinopharm/Beijing, Sputnik V -->> {'Nepal'}

            Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinovac, Sputnik V -->> {'North Macedonia'}

            Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac -->> {'Uruguay', 'Northern Cyprus'}

            CanSino, Covaxin, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinovac, Sputnik V -->> {'Pakistan'}

            Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sputnik Light, Sputnik V -->> {'Palestine'}

            Covaxin, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinovac, Sputnik V -->> {'Paraguay'}

            EpiVacCorona, Sputnik V -->> {'Russia'}

            Pfizer/BioNTech, Sputnik V -->> {'San Marino'}

            Oxford/AstraZeneca, Sinopharm/Beijing, Sputnik V -->> {'Seychelles'}

            Moderna, Pfizer/BioNTech, Sinovac -->> {'Singapore'}

            Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sputnik V -->> {'Slovakia'}

            Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sputnik V -->> {'Sri Lanka', 'Vietnam'}

            Johnson&Johnson, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinovac -->> {'Sudan'}

            Johnson&Johnson, Oxford/AstraZeneca, Sinopharm/Beijing, Sinovac, Sputnik Light, Sputnik V -->> {'Syria'}

            Medigen, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech -->> {'Taiwan'}

            Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac, Sputnik V -->> {'Tajikistan'}

            Johnson&Johnson -->> {'Tanzania'}

            EpiVacCorona, Oxford/AstraZeneca, Sinopharm/Beijing, Sputnik V -->> {'Turkmenistan'}

            Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinopharm/Wuhan, Sputnik V -->> {'United Arab Emirates'}

            Moderna, Oxford/AstraZeneca, Sputnik V, ZF2001 -->> {'Uzbekistan'}

            Abdala, Sinopharm/Beijing, Sinopharm/Wuhan, Sputnik V -->> {'Venezuela'}

            Johnson&Johnson, Oxford/AstraZeneca, Sinovac -->> {'Yemen'}
```
Finalmente, visualizamos estos datos para ver qué combinación de vacunas está usando cada país:
```python
in [8]:     import plotly.express as px
            import plotly.offline as py

            mapa_vacunacion = px.choropleth(data, locations = 'iso_code', color = 'vaccines')
            mapa_vacunacion.update_layout(height=300, margin={"r":0, "t":0, "l":0, "b":0})
            mapa_vacunacion.show()
```
![Mapa vacunación mundial](/img/posts/2021-05-11-analisis-vacunacion-covid19-python/2021-05-11-analisis-vacunacion-covid19-python-2.png)  

---

Así es como podemos analizar el tipo de vacunas que aplicó cada país a sus pobladores en la actualidad.  Podemos explorar más a profundidad este conjunto de datos, ya que hay mucho que puede hacer con estos datos.

Puedes descargar el cuaderno de Jupyter del proyecto [**aquí**](https://drive.google.com/file/d/1A6tf7CMc5X09JKXOL3TI5A-uOTMrV3nI/view?usp=sharing)
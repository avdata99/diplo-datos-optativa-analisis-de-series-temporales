# Análisis básico de series temporales

Primera optativa DiploDatos FAMAF.  
[Repo de las clases](https://github.com/DiploDatos/AnalisisSeriesTemporales).  

## Series temporales

[Clase 1](Series-Temporales.pdf).  

Queremos predecir la continuidad de los datos y definir márgenes de confianza para esa predicción.  
Una **diferencia grande** de las series temporales: los datos estan ordenados. No debo tomar una muestra aleatoria para analizar y dejar otros afuera como test.  
Muchas cosas cambias, esto no tiene que ver con lo anterior. Algunas cosas siguen siendo válidas (como _cross validation_ por ejemplo).  
LSTMs (Long Short-Term Memory Units, https://skymind.ai/wiki/lstm) incluye una secuencialidad de los datos y aplican acá.  

Cosas que se pronostican (o se intenta):
 - Valor de acciones y monedas
 - Demanda energética
 - Clima

Hay diferentes complejidades tratando de pronosticar series temporales. Saber a que hora va a salir el sol el año que viene es más facil de predecir que el clima que a su ves es más facil que el precio del dolar de la semana que viene.  

Los _lag plots_ sirven para detectar estacionalodad en los datos (lag 12 muestra cosas anuales).  

## Pedicción/Forecast

Ver slides

## Descomposicion

La series se descomponen en Estacionalidad, ciclos, tendencias y error. Los algoritmos pueden separar una línea de tiempo dibujada en otras donde cada una toma una parte.  


Case 2

**Autocorrelacion**. La relación entre una variable y sus valores anteriores (lags).  

**Expanding window** es ir cambiando gratualmente training y test (sobre la linea de tiempo)  
x test - train  
[--------------- xxxxxxxx]  
[----------------- xxxxxx]  
[------------------- xxxx]  

**Sliding window** es como la anterior pero se corre el punto de inicio. Se usa cuando la info más vieja no es relevante.  
[--------------- xxxxxxxx]  
[  --------------- xxxxxx]  
[    --------------- xxxx]  

**Precision** Medimos error. Por ejemplo root mean square error. Contra: son valores absolutos
Posible solucion: error porcentual. Ventaja: no depende de la escala de los valores de lo que analizo.  
**Errores escalados** es una opcion interesante.  

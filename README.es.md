# WeLoveReviews: sentimiento en reseñas de clientes

## Estado y datos necesarios

El archivo aportado en [data/raw/reviews.csv](data/raw/reviews.csv) contiene
500 reseñas utilizables, con columnas `review_id`, `rating` y `review_text`.
Se ejecutaron script y notebook completo desde kernel limpio, conservando las
500 filas y sus originales. No hay textos vacíos, IDs duplicados ni
truncamientos; se conservan siete duplicados adicionales de texto. Los 500
textos se detectaron como inglés. No se usaron datasets sustitutos.

Resultados del modelo: 412 positivas (82.4%), 41 neutrales (8.2%) y 47
negativas (9.4%); media predicha 4.398/5. Los ratings humanos de las mismas
filas promedian 4.5 y tienen 86.4% en 4-5 estrellas, no 90%. Hay 71 desacuerdos
exactos y 16 candidatos a falsos negativos por rating, no errores confirmados
automáticamente. La lectura de 20 ejemplos por el agente, asistida por IA,
está documentada; la validación humana sigue pendiente.

La inferencia está implementada en [src/app.py](src/app.py), y
[src/explore.ipynb](src/explore.ipynb) reutiliza esas funciones. No hace falta
PostgreSQL ni `DATABASE_URL` para este análisis. La revisión fija del modelo
`nlptown/bert-base-multilingual-uncased-sentiment` es
`8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d`, verificada mediante la API y los
archivos de esa revisión en Hugging Face. La carga del modelo/tokenizer y la
inferencia real han terminado correctamente, sin pesos en el repositorio.

## Reproducción

Entorno comprobado: Python 3.13; PyTorch 2.8.0 para CPU; Transformers 4.57.1.
Desde la raíz del repositorio:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
python -W error -m unittest discover -s tests -v
python src/app.py
python -m jupyter nbconvert --to notebook --execute --inplace src/explore.ipynb --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1800
```

El CSV aportado se reconoce sin parámetros: `review_text` para texto, `rating`
para estrellas humanas y `review_id` para identificador. Para otra exportación,
los nombres comunes se detectan de forma conservadora; si hay ambigüedad,
especifica los nombres exactos, por ejemplo con `--text-column`,
`--rating-column` y `--id-column`. Omite rating/ID si no existen. `--separator`
y `--encoding` permiten configurar la exportación. En el notebook configura
las mismas opciones en la celda inicial. No se presupone una escala humana
distinta de estrellas enteras 1-5.

El script funciona desde cualquier directorio usando su ruta absoluta; las
rutas relativas pasadas con `--input` y `--output` se resuelven respecto al
repositorio. Para ejecutar el notebook fuera de su árbol, define `WLR_ROOT`
como ruta absoluta del repositorio. El kernel de `nbconvert` debe corresponder
al Python donde se instalaron las dependencias.

## Salida y verificaciones

La salida real está en
[data/processed/reviews_with_sentiment.csv](data/processed/reviews_with_sentiment.csv).
Se conservan columnas originales, orden y duplicados. Se añaden `wl_fila`,
`wl_texto_limpio`, `wl_utilizable`, `wl_motivo`, `wl_idioma`,
`wl_estrellas_humanas`, `wl_tokens`, `wl_truncada`, `wl_estrellas`,
`wl_sentimiento`, `wl_score`, `wl_modelo` y `wl_revision`. Una colisión de
nombres se rechaza en lugar de sobrescribir columnas originales. Los IDs se
leen como texto; campos vacíos quedan vacíos y cadenas como `NA` se conservan.

Las filas sin texto permanecen sin predicción, con motivo explícito. Se
documenta cualquier diferencia respecto a las 500 utilizables esperadas.
El mapeo es 1-2 negativo, 3 neutral, 4-5 positivo. El modelo/tokenizer se
cargan una vez por ejecución, con lotes de 8 en CPU; `--batch-size` y
`--device` son configurables. No se entrena ni se guardan pesos en el repo.
La caché habitual de Hugging Face debe estar fuera del repositorio.

Solo se normalizan espacios, conservando negaciones y puntuación. Se cuentan
tokens originales y casos truncados por la derecha al límite comprobado del
tokenizer/modelo (512 incluyendo tokens especiales). El score no demuestra
corrección. El promedio declarado 4.5 es una referencia externa y no equivale
a 90% de reseñas positivas. Las comparaciones con rating humano usan las
mismas filas con estrellas válidas.

Se han verificado instalación/importaciones y seis pruebas en
[tests/test_app.py](tests/test_app.py): mapeo, filas/originales y duplicados,
CSV de ida y vuelta, selección reproducible y rechazo de columnas ambiguas o
cabeceras duplicadas. Los ejemplos mínimos de prueba son contratos aislados
con predictor simulado: no son datos de negocio, no descargan modelos ni
alimentan el análisis. La sexta prueba contrasta los CSV reales: originales,
orden, 500 filas utilizables con predicción, mapeo, scores y porcentajes.
Se comprobó previamente la independencia de rutas desde la raíz y `/tmp`;
ahora el script finalizó con los datos reales y produjo el CSV verificado.

Con datos reales, `validate_results` comprueba conservación, cobertura de las
utilizables, mapeo, rango de score y porcentajes; `verify_saved_csv` contrasta
el CSV con el notebook, con tolerancia absoluta de `1e-6` para scores.
Esas verificaciones de las 500 reseñas han pasado; el notebook coincide con
el CSV del script, tiene sus siete celdas ejecutadas y guarda salidas reales.

## Validación cualitativa pendiente

El agente leyó 20 ejemplos (10 aleatorios con semilla 42 y 10 focalizados),
con texto completo, interpretación asistida por IA, posible error y patrón.
La tabla conserva validación y comentario humano como pendientes. Por ejemplo,
ID 73 contiene "Service was fast and friendly" y deseo de volver, pero recibe
1 estrella predicha; ID 87 incluye "never disappoints" y adaptación a alergia,
también con 1 estrella. Son candidatos fuertes tras lectura del texto, sujetos
a validación humana. ID 322 es mixto y el cierre favorable puede quedar
infraponderado; ID 395 tiene espera de 25 minutos pese al balance positivo.

No se corrigieron predicciones ni se extrapoló tasa de error desde la muestra
focalizada. El desajuste entre productos y servicios, la negación y el balance
mixto son hipótesis apoyadas por ejemplos, no causas demostradas. Las
anotaciones son específicas de los textos leídos; si cambia la muestra, se
debe volver a revisar. Validar los 16 candidatos y los hallazgos cualitativos
antes de comunicar un porcentaje confirmado de clientes insatisfechos.

---

# Plantilla de Proyecto de Ciencia de Datos

Esta plantilla está diseñada para impulsar proyectos de ciencia de datos proporcionando una configuración básica para conexiones de base de datos, procesamiento de datos, y desarrollo de modelos de aprendizaje automático. Incluye una organización estructurada de carpetas para tus conjuntos de datos y un conjunto de paquetes de Python predefinidos necesarios para la mayoría de las tareas de ciencia de datos.

## Estructura

El proyecto está organizado de la siguiente manera:

- **`src/app.py`** → Script principal de Python donde correrá tu proyecto.
- **`src/explore.ipynb`** → Notebook para exploración y pruebas. Una vez finalizada la exploración, migra el código limpio a `app.py`.
- **`src/utils.py`** → Funciones auxiliares, como conexión a bases de datos.
- **`requirements.txt`** → Lista de paquetes de Python necesarios.
- **`models/`** → Contendrá tus clases de modelos SQLAlchemy.
- **`data/`** → Almacena los datasets en diferentes etapas:
  - **`data/raw/`** → Datos sin procesar.
  - **`data/interim/`** → Datos transformados temporalmente.
  - **`data/processed/`** → Datos listos para análisis.


## ⚡ Configuración Inicial en Codespaces (Recomendado)

No es necesario realizar ninguna configuración manual, ya que **Codespaces se configura automáticamente** con los archivos predefinidos que ha creado la academia para ti. Simplemente sigue estos pasos:

1. **Espera a que el entorno se configure automáticamente**.
   - Todos los paquetes necesarios y la base de datos se instalarán por sí mismos.
   - El `username` y `db_name` creados automáticamente están en el archivo **`.env`** en la raíz del proyecto.
2. **Una vez que Codespaces esté listo, puedes comenzar a trabajar inmediatamente**.


## 💻 Configuración en Local (Solo si no puedes usar Codespaces)

**Prerrequisitos**

Asegúrate de tener Python 3.11+ instalado en tu máquina. También necesitarás pip para instalar los paquetes de Python.

**Instalación**

Clona el repositorio del proyecto en tu máquina local.

Navega hasta el directorio del proyecto e instala los paquetes de Python requeridos:

```bash
pip install -r requirements.txt
```

**Crear una base de datos (si es necesario)**

Crea una nueva base de datos dentro del motor Postgres personalizando y ejecutando el siguiente comando: 

```bash
$ psql -U postgres -c "DO \$\$ BEGIN 
    CREATE USER mi_usuario WITH PASSWORD 'mi_contraseña'; 
    CREATE DATABASE mi_base_de_datos OWNER mi_usuario; 
END \$\$;"
```
Conéctate al motor Postgres para usar tu base de datos, manipular tablas y datos: 

```bash
$ psql -U mi_usuario -d mi_base_de_datos
```

¡Una vez que estés dentro de PSQL podrás crear tablas, hacer consultas, insertar, actualizar o eliminar datos y mucho más!

**Variables de entorno**

Crea un archivo .env en el directorio raíz del proyecto para almacenar tus variables de entorno, como tu cadena de conexión a la base de datos:

```makefile
DATABASE_URL="postgresql://<USUARIO>:<CONTRASEÑA>@<HOST>:<PUERTO>/<NOMBRE_BD>"

#example
DATABASE_URL="postgresql://mi_usuario:mi_contraseña@localhost:5432/mi_base_de_datos"
```

## Ejecutando la Aplicación

Para ejecutar la aplicación, ejecuta el script app.py desde la raíz del directorio del proyecto:

```bash
python src/app.py
```

## Añadiendo Modelos

Para añadir clases de modelos SQLAlchemy, crea nuevos archivos de script de Python dentro del directorio models/. Estas clases deben ser definidas de acuerdo a tu esquema de base de datos.

Definición del modelo de ejemplo (`models/example_model.py`):

```py
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

Base = declarative_base()

class ExampleModel(Base):
    __tablename__ = 'example_table'
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(unique=True)
```

## Trabajando con Datos

Puedes colocar tus conjuntos de datos brutos en el directorio data/raw, conjuntos de datos intermedios en data/interim, y los conjuntos de datos procesados listos para el análisis en data/processed.

Para procesar datos, puedes modificar el script app.py para incluir tus pasos de procesamiento de datos, utilizando pandas para la manipulación y análisis de datos.

## Contribuyentes

Esta plantilla fue construida como parte del [Data Science and Machine Learning Bootcamp](https://4geeksacademy.com/us/coding-bootcamps/datascience-machine-learning) de 4Geeks Academy por [Alejandro Sanchez](https://twitter.com/alesanchezr) y muchos otros contribuyentes. Descubre más sobre [los programas BootCamp de 4Geeks Academy](https://4geeksacademy.com/us/programs) aquí.

Otras plantillas y recursos como este se pueden encontrar en la página de GitHub de la escuela.
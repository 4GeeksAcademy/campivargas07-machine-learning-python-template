# WeLoveReviews: análisis de sentimiento en reseñas de clientes

## Resumen

Análisis de 500 reseñas de Harbor House Café para comparar el sentimiento
expresado en los textos con un promedio declarado de 4.5/5 estrellas.
Se integra un modelo existente, sin entrenamiento ni fine-tuning.

El artefacto principal de comunicación es
[src/explore.ipynb](src/explore.ipynb), con narrativa en español, tablas,
gráficos y salidas ejecutadas desde un kernel limpio. Este README sirve como
guía de revisión, no sustituye al análisis del notebook.

| Sentimiento predicho | Reseñas | Porcentaje |
| --- | ---: | ---: |
| Positivo | 412 | 82.4% |
| Neutral | 41 | 8.2% |
| Negativo | 47 | 9.4% |

La media predicha es **4.398/5**, frente a **4.5/5** en los ratings humanos
de las mismas 500 filas. El 86.4% de esos ratings es de 4–5 estrellas:
**4.5/5 no equivale a 90% de reseñas positivas**. Se conservaron todas las
filas, incluidos siete duplicados adicionales de texto. No hubo truncamientos.

## Qué Revisarán Los Profesores

| Criterio | Dónde encontrar la evidencia |
| --- | --- |
| Notebook ejecutado y arco narrativo claro | [src/explore.ipynb](src/explore.ipynb): objetivos, EDA, limpieza, plan/modelo, inferencia, resultados, revisión y conclusiones; Markdown entre bloques importantes. |
| Integración del modelo existente | [src/app.py](src/app.py): `from_pretrained()` y `pipeline()`, sin entrenamiento. |
| Modelo fijado y carga reutilizada | Modelo `nlptown/bert-base-multilingual-uncased-sentiment`, revisión SHA `8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d`; una carga por ejecución, reutilizada en lotes. |
| Cobertura de las 500 reseñas | [data/processed/reviews_with_sentiment.csv](data/processed/reviews_with_sentiment.csv): columnas originales y estrellas, banda de sentimiento y score por fila. |
| Distribución y comparación con 4.5 | Tablas, gráficos y conclusiones del notebook; comparación humana/modelo sobre las mismas filas. |
| Ejemplos revisados y posibles errores | Tabla de 20 reseñas: 10 aleatorias con semilla 42 y 10 focalizadas, con interpretación, posible error y patrón. |
| Candidatos a falsos negativos e hipótesis de dominio | 16 candidatos por discrepancia con rating; ejemplos leídos sobre personal, alergias, negaciones y opiniones mixtas. El desajuste entre productos y servicios es una hipótesis, no una causa demostrada. |
| Ruta de producción y resultados guardados | [src/app.py](src/app.py) genera el CSV; el notebook contrasta sus resultados con ese archivo. |
| Dependencias y pesos | [requirements.txt](requirements.txt) fija versiones exactas de las dependencias del análisis; [.gitignore](.gitignore) excluye cachés, entornos y pesos. Los pesos permanecen fuera del repositorio. |
| Verificaciones reproducibles | [tests/test_app.py](tests/test_app.py): seis pruebas de mapeo, conservación, serialización, selección y cobertura real. |

## Límites Y Pendientes

La lectura de los 20 ejemplos fue realizada por el agente **con asistencia de
IA, no por una persona**. Los campos de validación y comentario humano siguen
pendientes. Para cumplir literalmente la revisión manual humana de la rúbrica,
el estudiante debe leer los textos y completar esos campos antes de entregar.
Una discrepancia con rating no confirma por sí sola un error, y la muestra
focalizada no estima la tasa poblacional de errores.

El score no garantiza corrección. No se corrigieron automáticamente las
predicciones ni se interpreta el 9.4% negativo como porcentaje confirmado de
clientes insatisfechos. La procedencia y el periodo del promedio declarado
de la plataforma no se verifican solo porque coincida con la media del CSV.

Las dependencias necesarias para el análisis están fijadas exactamente;
algunos paquetes heredados de la plantilla mantienen rangos o no tienen
versión. La comparación con `tabularisai/multilingual-sentiment-analysis` es
opcional y no forma parte de esta ejecución.

## Reproducir Y Entregar

Entorno comprobado: Python 3.13 y PyTorch para CPU. Desde la raíz:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
python src/app.py
python -m jupyter nbconvert --to notebook --execute --inplace src/explore.ipynb --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1800
python -W error -m unittest discover -s tests -v
```

No se necesita PostgreSQL ni `DATABASE_URL` para WeLoveReviews. Consulta
[README.es.md](README.es.md) para opciones de columnas, rutas y ejecución.

Antes de entregar, completar la validación humana y publicar los cambios en el
repositorio propio de GitHub. Incluir el notebook con salidas, el script y el
CSV procesado; no incluir pesos ni entornos. Entregar el enlace siguiendo el
proceso del instructor. La publicación no se realiza automáticamente.

---

## Documentación Original De La Plantilla

# Data Science Project Boilerplate

This boilerplate is designed to kickstart data science projects by providing a basic setup for database connections, data processing, and machine learning model development. It includes a structured folder organization for your datasets and a set of pre-defined Python packages necessary for most data science tasks.

## Structure

The project is organized as follows:

- **`src/app.py`** → Main Python script where your project will run.
- **`src/explore.ipynb`** → Notebook for exploration and testing. Once exploration is complete, migrate the clean code to `app.py`.
- **`src/utils.py`** → Auxiliary functions, such as database connection.
- **`requirements.txt`** → List of required Python packages.
- **`models/`** → Will contain your SQLAlchemy model classes.
- **`data/`** → Stores datasets at different stages:
  - **`data/raw/`** → Raw data.
  - **`data/interim/`** → Temporarily transformed data.
  - **`data/processed/`** → Data ready for analysis.


## ⚡ Initial Setup in Codespaces (Recommended)

No manual setup is required, as **Codespaces is automatically configured** with the predefined files created by the academy for you. Just follow these steps:

1. **Wait for the environment to configure automatically**.
   - All necessary packages and the database will install themselves.
   - The automatically created `username` and `db_name` are in the **`.env`** file at the root of the project.
2. **Once Codespaces is ready, you can start working immediately**.


## 💻 Local Setup (Only if you can't use Codespaces)

**Prerequisites**

Make sure you have Python 3.11+ installed on your machine. You will also need pip to install the Python packages.

**Installation**

Clone the project repository to your local machine.

Navigate to the project directory and install the required Python packages:

```bash
pip install -r requirements.txt
```

**Create a database (if necessary)**

Create a new database within the Postgres engine by customizing and executing the following command:

```bash
$ psql -U postgres -c "DO \$\$ BEGIN 
    CREATE USER my_user WITH PASSWORD 'my_password'; 
    CREATE DATABASE my_database OWNER my_user; 
END \$\$;"
```
Connect to the Postgres engine to use your database, manipulate tables, and data:

```bash
$ psql -U my_user -d my_database
```

Once inside PSQL, you can create tables, run queries, insert, update, or delete data, and much more!

**Environment Variables**

Create a .env file in the root directory of the project to store your environment variables, such as your database connection string:

```makefile
DATABASE_URL="postgresql://<USER>:<PASSWORD>@<HOST>:<PORT>/<DB_NAME>"

#example
DATABASE_URL="postgresql://my_user:my_password@localhost:5432/my_database"
```

## Running the Application

To run the application, execute the app.py script from the root directory of the project:

```bash
python src/app.py
```

## Adding Models

To add SQLAlchemy model classes, create new Python script files within the models/ directory. These classes should be defined according to your database schema.

Example model definition (`models/example_model.py`):

```py
from sqlalchemy.orm import declarative_base
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

Base = declarative_base()

class ExampleModel(Base):
    __tablename__ = 'example_table'
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(unique=True)
```

## Working with Data

You can place your raw datasets in the data/raw directory, intermediate datasets in data/interim, and processed datasets ready for analysis in data/processed.

To process data, you can modify the app.py script to include your data processing steps, using pandas for data manipulation and analysis.

## Contributors

This template was built as part of the [Data Science and Machine Learning Bootcamp](https://4geeksacademy.com/us/coding-bootcamps/datascience-machine-learning) by 4Geeks Academy by [Alejandro Sanchez](https://twitter.com/alesanchezr) and many other contributors. Learn more about [4Geeks Academy BootCamp programs](https://4geeksacademy.com/us/programs) here.

Other templates and resources like this can be found on the school's GitHub page.
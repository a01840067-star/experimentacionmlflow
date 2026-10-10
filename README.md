# ExperimentacionMLFlow

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Script de entrenamiento parametrizado para predecir mpg (Auto MPG, UCI) con registro de experimentos monitoreados en MLFlow

## Requisitos
- Probado en Linux (Ubuntu en WSL 2) con Miniconda (conda 26.7.1) y Python 3.11
- Git

## Instalacion 
```bash
git clone https://github.com/a01840067-star/experimentacionmlflow.git
```
```bash
cd experimentacionmlflow
```
```bash
conda env create -f environment.yml
```
```bash
conda activate experimentacionmlflow
```

## Reproducibilidad
La semilla (`--seed`, default 42) controla el split train/test; el resto del flujo es determinista, por lo que dos ejecuciones con los mismos argumentos producen las mismas metricas. Las versiones de las dependencias estan fijadas en `environment.yml`.

## Ejecucion
Todos los comandos se ejecutan desde la raiz del proyecto
```bash
python -m experimentacionmlflowModule.modeling.train
```
Con hiperparametros personalizados: 
```bash
python -m experimentacionmlflowModule.modeling.train --degree 2 --alpha 10 --log-target
```

| Argumento | Default | Descripcion |
|---|---|---|
| `--alpha` | `1.0` | Fuerza de regularizacion de Ridge |
| `--degree` | `1` | Grado de PolynomialFeatures (1 = lineal) |
| `--log-target` | desactivado | Entrena sobre log(mpg) |
| `--test-size` | `0.2` | Proporcion del conjunto de prueba |
| `--seed` | `42` | Semilla aleatoria del split |
| `--experiment` | `<nombre del experimento>` | Experimento de MLflow |

Para ver la ayuda completa: `python -m experimentacionmlflowModule.modeling.train --help`

Para repetir todas las corridas del analisis (requiere bash: Linux, macOS o WSL):
```bash
./run_experiments.sh
```
## Ver resultados en MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```


--------


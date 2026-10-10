import argparse
import json
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from experimentacionmlflowModule.config import FIGURES_DIR, PROJ_ROOT, RANDOM_STATE
from experimentacionmlflowModule.dataset import cargar_datos
from experimentacionmlflowModule.features import preparar_datos, dividir, imputar_horsepower
from experimentacionmlflowModule.plots import graficar_residuos

def parse_args() -> argparse.Namespace:
    """Lee los hiperparametros y la configuracion desde la linea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrena un modelo Ridge para predecir mpg (Auto MPG) y registra la corrida en MLflow."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Fuerza de regularizacion de Ridge (default: 1.0)",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=1,
        help="Grado de PolynomialFeatures; 1 = modelo lineal (default: 1)",
    )
    parser.add_argument(
        "--log-target",
        action="store_true",
        help="Entrena sobre log(mpg) en lugar de mpg (default: desactivado)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporcion del conjunto de prueba (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help=f"Semilla aleatoria para el split (default: {RANDOM_STATE})",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="auto-mpg-ridge",
        help="Nombre del experimento en MLflow (default: auto-mpg-ridge)",
    )
    return parser.parse_args()

def construir_modelo(alpha: float, degree: int) -> Pipeline:
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )

def evaluar(modelo: Pipeline, X, y, log_target: bool) -> dict:
    pred = modelo.predict(X)
    if log_target:
        pred = np.exp(pred)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }

def main() -> None:

    args = parse_args()
    np.random.seed(args.seed)

    # 1. Datos: carga, preparacion, split e imputacion
    X, y = preparar_datos(cargar_datos())
    X_train, X_test, y_train, y_test = dividir(X, y, args.test_size, args.seed)
    X_train, X_test = imputar_horsepower(X_train, X_test)

    # 2. Configuracion de MLflow
    mlflow.set_tracking_uri(f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(args.experiment)

    run_name = f"ridge_a{args.alpha}_d{args.degree}" + ("_log" if args.log_target else "")

    with mlflow.start_run(run_name=run_name):
        # 3. Entrenamiento
        modelo = construir_modelo(args.alpha, args.degree)
        y_fit = np.log(y_train) if args.log_target else y_train
        modelo.fit(X_train, y_fit)

        # 4. Parametros
        mlflow.log_param("alpha", args.alpha)
        mlflow.log_param("degree", args.degree)
        mlflow.log_param("log_target", args.log_target)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))

        # 5. Metricas en train y test
        metricas = {
            "train": evaluar(modelo, X_train, y_train, args.log_target),
            "test": evaluar(modelo, X_test, y_test, args.log_target),
        }
        for conjunto, valores in metricas.items():
            for nombre, valor in valores.items():
                mlflow.log_metric(f"{conjunto}_{nombre}", valor)

        # 6. Artefacto: grafico de residuos
        pred_test = modelo.predict(X_test)
        if args.log_target:
            pred_test = np.exp(pred_test)
        ruta_grafico = graficar_residuos(y_test, pred_test, FIGURES_DIR / "residuos_test.png")
        mlflow.log_artifact(str(ruta_grafico))

        # 7. Artefacto: archivo de metricas
        ruta_metricas = FIGURES_DIR.parent / "metricas.json"
        ruta_metricas.write_text(json.dumps(metricas, indent=2))
        mlflow.log_artifact(str(ruta_metricas))

        # 8. Modelo entrenado
        mlflow.set_tag("dataset", "UCI Auto MPG (id 9)")
        mlflow.sklearn.log_model(modelo, name="modelo", input_example=X_train.head(3))

        print(f"Corrida '{run_name}' -> test RMSE: {metricas['test']['rmse']:.3f}")


if __name__ == "__main__":
    main()
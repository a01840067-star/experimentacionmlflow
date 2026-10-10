from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

def graficar_residuos(y_test, pred_test, ruta: Path) -> Path:
    residuos = y_test - pred_test

    fig_residuos, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test, pred_test, alpha=0.7)
    lims = [y_test.min() - 2, y_test.max() + 2]
    axes[0].plot(lims, lims, 'r--')
    axes[0].set_xlabel('mpg real')
    axes[0].set_ylabel('mpg predicho')
    axes[0].set_title('Real frente a predicho (test)')

    axes[1].scatter(pred_test, residuos, alpha=0.7)
    axes[1].axhline(0, color='r', linestyle='--')
    axes[1].set_xlabel('mpg predicho')
    axes[1].set_ylabel('Residuo (real | predicho)')
    axes[1].set_title('Residuos (test)')

    fig_residuos.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig_residuos.savefig(ruta, dpi=120, bbox_inches="tight")
    plt.close(fig_residuos)
    return ruta
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# === Cargar el archivo CSV ===
file_path = "results.csv"  # cambia la ruta si está en otra carpeta
results = pd.read_csv(file_path)

# === Crear archivo PDF con las tres gráficas ===
pdf_path = "yolov11_results_dark.pdf"

with PdfPages(pdf_path) as pdf:
    # --- 1. Pérdidas de entrenamiento ---
    plt.figure(figsize=(8,5))
    plt.plot(results["epoch"], results["train/box_loss"], label="Box Loss", color="#1f77b4")
    plt.plot(results["epoch"], results["train/cls_loss"], label="Cls Loss", color="#2ca02c")
    plt.plot(results["epoch"], results["train/dfl_loss"], label="DFL Loss", color="#d62728")
    plt.xlabel("Época")
    plt.ylabel("Valor de pérdida")
    plt.title("Pérdidas de entrenamiento por época")
    plt.legend()
    plt.grid(True)  # grid activado
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # --- 2. Métricas de evaluación ---
    plt.figure(figsize=(8,5))
    plt.plot(results["epoch"], results["metrics/precision(B)"], label="Precisión (B)", color="#9467bd")
    plt.plot(results["epoch"], results["metrics/recall(B)"], label="Recall (B)", color="#8c564b")
    plt.plot(results["epoch"], results["metrics/mAP50(B)"], label="mAP50 (B)", color="#17becf")
    plt.plot(results["epoch"], results["metrics/mAP50-95(B)"], label="mAP50-95 (B)", color="#ff7f0e")
    plt.xlabel("Época")
    plt.ylabel("Valor de métrica")
    plt.title("Métricas de evaluación por época")
    plt.legend()
    plt.grid(True)  # grid activado
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # --- 3. Tasas de aprendizaje ---
    plt.figure(figsize=(8,5))
    plt.plot(results["epoch"], results["lr/pg0"], label="lr/pg0", color="#1f77b4")
    plt.plot(results["epoch"], results["lr/pg1"], label="lr/pg1", color="#ff7f0e")
    plt.plot(results["epoch"], results["lr/pg2"], label="lr/pg2", color="#2ca02c")
    plt.xlabel("Época")
    plt.ylabel("Learning Rate")
    plt.title("Tasas de aprendizaje por época")
    plt.legend()
    plt.grid(True)  # grid activado
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print(f"PDF generado correctamente: {pdf_path}")

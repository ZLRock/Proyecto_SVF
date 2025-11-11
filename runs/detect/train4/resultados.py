import pandas as pd
import matplotlib.pyplot as plt

# Leer el archivo
df = pd.read_csv("results.csv")

# Asegurarse de que existe la columna epoch (algunos CSV no la incluyen)
if 'epoch' not in df.columns:
    df.insert(0, 'epoch', range(1, len(df)+1))

# ====== FIGURA 1: PÉRDIDAS ======
plt.figure(figsize=(10,6))
plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
plt.plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss')
plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
plt.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.title('Evolución de pérdidas durante el entrenamiento')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('losses.png', dpi=300)
plt.close()

# ====== FIGURA 2: MÉTRICAS ======
plt.figure(figsize=(10,6))
plt.plot(df['epoch'], df['metrics/precision'], label='Precision')
plt.plot(df['epoch'], df['metrics/recall'], label='Recall')
plt.plot(df['epoch'], df['metrics/mAP50'], label='mAP@50')
plt.plot(df['epoch'], df['metrics/mAP50-95'], label='mAP@50-95')
plt.xlabel('Época')
plt.ylabel('Valor')
plt.title('Métricas de validación')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('metrics.png', dpi=300)
plt.close()

# ====== FIGURA 3: LEARNING RATE ======
lr_cols = [c for c in df.columns if c.startswith('lr/')]
if lr_cols:
    plt.figure(figsize=(8,5))
    for c in lr_cols:
        plt.plot(df['epoch'], df[c], label=c)
    plt.xlabel('Época')
    plt.ylabel('Learning Rate')
    plt.title('Tasa de aprendizaje')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('learning_rate.png', dpi=300)
    plt.close()

print("✅ Gráficas generadas: losses.png, metrics.png, learning_rate.png")

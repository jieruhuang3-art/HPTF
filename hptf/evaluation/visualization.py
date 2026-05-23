import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(matrix, labels, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

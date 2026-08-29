"""Create presentation-ready evaluation charts from the grouped training reports."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
REPORTS.mkdir(exist_ok=True)
comparison = pd.read_csv(REPORTS / 'model_comparison_augmented.csv')
summary = json.loads((REPORTS / 'training_summary_augmented.json').read_text(encoding='utf-8'))
data = pd.read_csv(ROOT / 'data' / 'processed_features_augmented.csv')

plt.style.use('seaborn-v0_8-whitegrid')
labels = comparison['model'].str.replace('_', ' ').str.title()
x = np.arange(len(labels))
metrics = [('test_accuracy', 'Accuracy'), ('test_precision', 'Precision'), ('test_recall', 'Recall'), ('test_f1', 'F1 score'), ('cv_f1_mean', 'CV F1')]
fig, ax = plt.subplots(figsize=(11, 6))
width = 0.15
for index, (column, label) in enumerate(metrics):
    bars = ax.bar(x + (index - 2) * width, comparison[column] * 100, width, label=label)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .6, f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)
ax.set_title('Model Performance on Leakage-Safe Grouped Evaluation', weight='bold')
ax.set_ylabel('Score (%)')
ax.set_ylim(0, 105)
ax.set_xticks(x, labels, rotation=12)
ax.legend(ncol=5, loc='lower center', bbox_to_anchor=(.5, -0.25))
fig.tight_layout()
fig.savefig(REPORTS / 'model_performance_comparison.png', dpi=200, bbox_inches='tight')
plt.close(fig)

best = summary['selected_model']
best_result = next(item for item in summary['results'] if item['model'] == best)
cm = np.array(best_result['confusion_matrix'])
fig, ax = plt.subplots(figsize=(6, 5))
image = ax.imshow(cm, cmap='Blues')
fig.colorbar(image, ax=ax, label='Number of snippets')
ax.set(title=f'Confusion Matrix: {best.replace("_", " ").title()}', xlabel='Predicted label', ylabel='Actual label', xticks=[0, 1], yticks=[0, 1], xticklabels=['Clean', 'Buggy'], yticklabels=['Clean', 'Buggy'])
threshold = cm.max() / 2
for row in range(2):
    for col in range(2):
        ax.text(col, row, str(cm[row, col]), ha='center', va='center', fontsize=16, color='white' if cm[row, col] > threshold else 'black')
fig.tight_layout()
fig.savefig(REPORTS / 'best_model_confusion_matrix.png', dpi=200, bbox_inches='tight')
plt.close(fig)

counts = data['label'].value_counts().reindex([0, 1], fill_value=0)
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(['Clean', 'Buggy'], counts.values, color=['#4C78A8', '#E45756'])
for bar, count in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, count + 15, f'{count:,}', ha='center', weight='bold')
ax.set_title('Augmented Training Dataset Label Distribution', weight='bold')
ax.set_ylabel('Number of snippets')
ax.set_ylim(0, counts.max() * 1.12)
fig.tight_layout()
fig.savefig(REPORTS / 'augmented_dataset_label_distribution.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print('Created:', ', '.join(path.name for path in [REPORTS / 'model_performance_comparison.png', REPORTS / 'best_model_confusion_matrix.png', REPORTS / 'augmented_dataset_label_distribution.png']))

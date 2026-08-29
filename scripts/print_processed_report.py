import json
import pandas as pd

def main():
    df = pd.read_csv('data/processed_features.csv')
    out = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'label_distribution': df['label'].value_counts().to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_describe': df.select_dtypes(include=['number']).describe().to_dict(),
        'n_malformed_by_ast': int((df['ast_parse_success'] == 0).sum()),
    }
    print(json.dumps(out, indent=2, default=str))

if __name__ == '__main__':
    main()

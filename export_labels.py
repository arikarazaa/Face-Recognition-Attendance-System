"""
Run this once after training to export label names to labels.json.
Used by any front-end or external tool that needs the class list.
"""
import json
import joblib


def export(encoder_path='label_encoder.pkl', output_path='labels.json'):
    encoder = joblib.load(encoder_path)
    labels  = list(encoder.classes_)

    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"✅ Exported {len(labels)} labels to '{output_path}'")
    print(f"   Labels: {labels}")


if __name__ == "__main__":
    export()
import joblib

def inspect_label_encoder(encoder_path='label_encoder.pkl'):
    label_encoder = joblib.load(encoder_path)
    print("Label Encoder Classes:")
    print(label_encoder.classes_)

inspect_label_encoder()

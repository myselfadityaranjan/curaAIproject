import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

def recreate_and_save_label_encoder(labels_path='labels.npy', encoder_path='label_encoder.pkl'): #load labls from .npy file which holds labels from dataset directories
    # load original labels
    labels = np.load(labels_path)
    
    # create LabelEncoder to fit to loaded labels
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    
    # save LabelEncoder to a .pkl file
    joblib.dump(label_encoder, encoder_path)
    print(f"LabelEncoder saved to {encoder_path}")

if __name__ == "__main__":
    recreate_and_save_label_encoder()

from flask import Flask, request, jsonify, render_template
import torch
from torchvision import models, transforms
from PIL import Image
import joblib

app = Flask(__name__)

# Load your model and label encoder
def load_model(model_path='model.pth', num_classes=8):
    print("Loading model...")
    
    # Define the model architecture (e.g., ResNet18)
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Adjust the output layer to match the number of classes
    
    # Load the state_dict (the weights of the model)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully.")
    
    return model

def load_label_encoder(encoder_path='label_encoder.pkl'):
    print("Loading label encoder...")
    label_encoder = joblib.load(encoder_path)
    print("Label encoder loaded successfully.")
    return label_encoder

# Initialize the model and label encoder
model = load_model('model.pth', num_classes=8)  # Adjust the number of classes as necessary
label_encoder = load_label_encoder('label_encoder.pkl')

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def predict_image(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
    return outputs

def get_label_name(predicted_class_index):
    return label_encoder.inverse_transform([predicted_class_index])[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        image_path = 'temp.jpg'
        file.save(image_path)
        
        image_tensor = preprocess_image(image_path)
        outputs = predict_image(image_tensor)
        
        predicted_class_index = torch.argmax(outputs, 1).item()
        predicted_label_name = get_label_name(predicted_class_index)
        
        return jsonify({'predicted_label': predicted_label_name})

if __name__ == "__main__":
    app.run(debug=True)

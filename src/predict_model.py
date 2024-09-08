import torch
import torchvision.transforms as transforms
from PIL import Image
import joblib
from torchvision import models
import torch.nn.functional as F

def load_model(model_path='model.pth'): #load model checkpoint from .pth file on disk
    print("Loading model...")
    checkpoint = torch.load(model_path)

    num_classes = 8  # manually input number of output classes

    # initialize ResNet model
    model = models.resnet18(weights=None)  # we are not using the pretrained weights
    num_ftrs = model.fc.in_features #adjust final layer to match output classes
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    
    # load the model state dict
    model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded successfully.")
    return model

def load_label_encoder(encoder_path='label_encoder.pkl'): #load label encoader during training
    print("Loading label encoder...")
    label_encoder = joblib.load(encoder_path)
    print("Label encoder loaded successfully.")
    return label_encoder

def preprocess_image(image_path):
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert('RGB') #ensure 3 channel image 
    
    transform = transforms.Compose([ #transformations to images
        transforms.Resize((224, 224)),     # resize image to 224x224
        transforms.ToTensor(),             # convert PIL Image to tensor (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # add batch dimension
    return image_tensor

def predict_image(model, image_path):
    image_tensor = preprocess_image(image_path) #preprocess image before feeding to model
    with torch.no_grad():
        outputs = model(image_tensor)
    return outputs

def get_label_name(predicted_class_index, label_encoder): #convert predicted class index to actual class name
    return label_encoder.inverse_transform([predicted_class_index])[0]

def main():
    # path to the image
    image_path = '/Users/adityaranjan/Documents/CURA AI/test image/brainglioma.png'  # can be anything; this is just my practice image
    
    # load the model and label encoder
    model = load_model()
    label_encoder = load_label_encoder()

    # predict
    outputs = predict_image(model, image_path)
    print(f"Raw model outputs: {outputs}")

    predicted_class_index = torch.argmax(outputs, 1).item()
    print(f"Predicted class index: {predicted_class_index}")

    predicted_label_name = get_label_name(predicted_class_index, label_encoder)
    print(f"Predicted class for the image is: {predicted_label_name}")

if __name__ == "__main__":
    main()

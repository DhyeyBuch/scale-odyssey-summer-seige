import argparse
import torch
import torch.nn.functional as F
from PIL import Image

from config import CLASS_NAMES, BEST_MODEL_PATH
from model import load_model
from dataset import val_test_transform


def predict(image_path, model, device):
    img    = Image.open(image_path).convert('RGB')
    tensor = val_test_transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs  = F.softmax(output, dim=1)[0]

    predicted_idx  = probs.argmax().item()
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probs[predicted_idx].item()

    print(f"Prediction : {predicted_class}")
    print(f"Confidence : {confidence:.1%}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify an astronomical image')
    parser.add_argument('--image', required=True, help='Path to the image file')
    
    parser.add_argument('--checkpoint', default=str(BEST_MODEL_PATH),
                        help='Path to model checkpoint (default: BEST_MODEL_PATH from config)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(args.checkpoint, device)

    predict(args.image, model, device)
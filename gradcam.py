import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from config import CLASS_NAMES, BEST_MODEL_PATH, GRADCAM_LAYER, GRADCAM_ALPHA
from model import load_model
from dataset import val_test_transform


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for ResNet-50.

    Hooks are registered on layer4 (the final convolutional block).
    This gives a 7x7 spatial map of which regions most influenced
    the predicted class score, which is then upsampled to 224x224.

    """
    def __init__(self, model):
        self.model      = model
        self.gradients  = None
        self.activations = None
        self._handles   = []

        self._handles.append(
            self.model.layer4.register_forward_hook(self._save_activations)
        )
        self._handles.append(
            self.model.layer4.register_full_backward_hook(self._save_gradients)
        )

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        """
        To Prevent accumulation of hooks, we have to remove them timely
        """
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def generate(self, input_tensor, target_class):
        """
        input_tensor : preprocessed image tensor (1, 3, 224, 224) on device
        target_class : integer class index to explain
        returns      : numpy array (224, 224)
        """
        self.model.eval()

        output = self.model(input_tensor)
        self.model.zero_grad()

        # Backpropagate only through the target class score
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1.0
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)

        # Upsampling the 7x7 to 224x224
        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def visualise_gradcam(model, image_path, device, target_class=None, save_dir=None):
    """
    Runs GradCAM on a single image and plots four panels:
      1. Original image
      2. Raw GradCAM heatmap
      3. Heatmap overlaid on original
      4. Class probability bar chart
    """
    img        = Image.open(image_path).convert('RGB')
    img_tensor = val_test_transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output          = model(img_tensor)
        probs           = F.softmax(output, dim=1)[0]
        predicted_class = output.argmax(dim=1).item()
        confidence      = probs[predicted_class].item()

    explain_class = target_class if target_class is not None else predicted_class

    gradcam = GradCAM(model)
    cam     = gradcam.generate(img_tensor, explain_class)
    gradcam.remove_hooks()

    # Building the overlay
    img_resized = img.resize((224, 224))
    img_array   = np.array(img_resized) / 255.0
    heatmap     = cm.jet(cam)[:, :, :3]
    overlay     = np.clip(
        (1 - GRADCAM_ALPHA) * img_array + GRADCAM_ALPHA * heatmap, 0, 1
    )

    # Plot showing the original image along with the heatmap overlap and all class probs
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(img_resized)
    axes[0].set_title('Original', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title(
        f'GradCAM\n(explaining: {CLASS_NAMES[explain_class]})', fontsize=12
    )
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(
        f'Overlay\nPredicted: {CLASS_NAMES[predicted_class]} ({confidence:.1%})',
        fontsize=12
    )
    axes[2].axis('off')

    # 4th panel in the plot showing Probability for all classes
    colors = ['#e74c3c' if i == predicted_class else '#3498db'
              for i in range(len(CLASS_NAMES))]
    axes[3].barh(CLASS_NAMES, probs.cpu().numpy(), color=colors)
    axes[3].set_xlim(0, 1)
    axes[3].set_title('Class Probabilities', fontsize=12)
    for i, v in enumerate(probs.cpu().numpy()):
        axes[3].text(v + 0.01, i, f'{v:.1%}', va='center', fontsize=9)

    plt.suptitle(os.path.basename(image_path), fontsize=10, y=1.02)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'gradcam_{os.path.basename(image_path)}')
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run GradCAM on an astronomical image or a folder of images'
    )
    parser.add_argument('--image',
                        help='Path to a single image')

    parser.add_argument('--true-class',
                        help='True class name — used to explain why the correct '
                             'class lost on misclassified images')
    parser.add_argument('--checkpoint', default=str(BEST_MODEL_PATH),
                        help='Path to model checkpoint (default: BEST_MODEL_PATH from config)')
    args = parser.parse_args()

    if not args.image:
        parser.error('Provide --image')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(args.checkpoint, device)

    if args.image:
        target_class = CLASS_NAMES.index(args.true_class) if args.true_class else None
        visualise_gradcam(model, args.image, device, target_class=target_class)

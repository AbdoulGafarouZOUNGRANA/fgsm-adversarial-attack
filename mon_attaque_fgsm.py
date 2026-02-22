import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files
import io
import requests

# 1. Model Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.to(device)
model.eval()

# 2. Image Upload
print("Upload the input image (Astronaut or Scuba Diver recommended):")
uploaded = files.upload()
filename = next(iter(uploaded))
img = Image.open(io.BytesIO(uploaded[filename])).convert('RGB')

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
img_tensor = preprocess(img).unsqueeze(0).to(device)
img_tensor.requires_grad = True

# 3. FGSM Attack Logic
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# 4. Inference (Before Attack)
output = model(img_tensor)
init_pred_idx = output.max(1, keepdim=True)[1]

# Load ImageNet labels
lab_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(lab_url).text.split('\n')
label_orig = labels[init_pred_idx.item()]

# Calculate Gradient
criterion = nn.CrossEntropyLoss()
loss = criterion(output, init_pred_idx[0])
model.zero_grad()
loss.backward()
data_grad = img_tensor.grad.data

# 5. Execute Attack
epsilon = 0.05  # Perturbation magnitude
perturbed_data = fgsm_attack(img_tensor, epsilon, data_grad)

# Inference (After Attack)
final_output = model(perturbed_data)
final_pred_idx = final_output.max(1, keepdim=True)[1]
label_adv = labels[final_pred_idx.item()]

# 6. Visualization (Scientific Style)
plt.figure(figsize=(12, 5))

# Original Plot
plt.subplot(1, 3, 1)
plt.imshow(img_tensor.squeeze().detach().cpu().permute(1, 2, 0))
plt.title(f"Original Prediction:\n{label_orig}", fontsize=10, fontweight='bold')
plt.axis('off')

# Noise Plot
plt.subplot(1, 3, 2)
noise = (data_grad.sign()).squeeze().detach().cpu().permute(1, 2, 0)
noise = (noise - noise.min()) / (noise.max() - noise.min())
plt.imshow(noise, cmap='gray')
plt.title("Adversarial Perturbation\n(Gradient Sign)", fontsize=10)
plt.axis('off')

# Adversarial Plot
plt.subplot(1, 3, 3)
plt.imshow(perturbed_data.squeeze().detach().cpu().permute(1, 2, 0))
plt.title(f"Adversarial Prediction:\n{label_adv}", fontsize=10, fontweight='bold', color='red')
plt.axis('off')

plt.tight_layout()
plt.show()
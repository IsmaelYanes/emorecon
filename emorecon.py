import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import LinearLR

#Directorio y parámetros iniciales
DATA_DIR = os.path.expanduser('ruta de data')
BATCH_SIZE = 64
IMG_SIZE = (48, 48)
EPOCHS = 20
LEARNING_RATE = 0.0004
SEED = 42

#En caso de tener GPU no usar CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED) #Generación por semilla

#Transformación del formato de imágenes
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

#Cargar datos de entrenamiento, validación y prueba
train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'val'), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'test'), transform=transform)

#Carga de datos por lotes
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#Transfer learning con resnet18
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 5)
model = model.to(device)

#Función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08)
scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=int(EPOCHS * 0.1)) #Reduce el learning rate por épocas

#Listas para almacenar las métricas
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

#Función evaluación del modelo
def evaluate(model, loader, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    loss = running_loss / len(loader)
    return loss, accuracy, all_labels, all_preds

#Entrenamiento
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_start_time = time.time() #Tiempo de ejecución por época

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_accuracy = 100 * correct / total
    epoch_loss = running_loss / len(train_loader)
    val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion)
    epoch_time = time.time() - epoch_start_time

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    scheduler.step() #cambiar learning rate

    #Estadísticas de época
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Time: {epoch_time:.2f}s")

#Evaluación de conjunto de prueba
test_loss, test_accuracy, test_labels, test_preds = evaluate(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

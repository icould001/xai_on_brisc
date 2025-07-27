import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from tqdm import tqdm
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from torchmetrics import MetricCollection

import json
import os

def get_resnet_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.BatchNorm1d(model.fc.in_features),
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.3),
        nn.Linear(64, num_classes)
    )
    fc_params = list(model.fc.parameters())
    fc_param_ids = set(map(id, fc_params))
    backbone_params = [p for p in model.parameters() if id(p) not in fc_param_ids]

    params_to_update = [
        {'params': backbone_params, 'lr': 1e-4},
        {'params': fc_params, 'lr': 1e-3}
    ]
    return model, params_to_update, "layer4", "resnet50"

def get_densenet_model(num_classes):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(model.classifier.in_features),
        nn.Dropout(0.3),
        nn.Linear(model.classifier.in_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.3),
        nn.Linear(64, num_classes)
    )
    fc_params = list(model.classifier.parameters())
    fc_param_ids = set(map(id, fc_params))
    backbone_params = [p for p in model.parameters() if id(p) not in fc_param_ids]


    params_to_update = [
        {'params': backbone_params, 'lr': 1e-4},
        {'params': fc_params, 'lr': 1e-3}
    ]

    return model, params_to_update, "features.denseblock4", "densenet121"

def train(model_func, epochs, train_loader, val_loader):
    num_classes = len(train_loader.dataset.classes)
    model, params_to_update, layer_name, model_name = model_func(num_classes)
    device = torch.device('cuda')
    model.to(device)
    # === Loss and Optimizer ===
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params_to_update)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    metrics = MetricCollection({
        'accuracy': Accuracy(task='multiclass', num_classes=num_classes),
        'f1': F1Score(task='multiclass', num_classes=num_classes, average='macro'),
        'precision': Precision(task='multiclass', num_classes=num_classes, average='macro'),
        'recall': Recall(task='multiclass', num_classes=num_classes, average='macro')
    }).to(device)

    train_stats = {
        'epoch': [],
        'loss': [],
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': []
    }
    
    val_stats = {
        'epoch': [],
        'loss': [],
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': []
    }

    # === Training Loop ===
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        metrics.reset()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)

            # Update metrics
            metrics.update(preds, labels)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    
        train_acc = correct / total
        train_loss = running_loss / total
        train_metrics = metrics.compute()
        print(f"Train Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f} | "
            f"Precision: {train_metrics['precision']:.4f} | Recall: {train_metrics['recall']:.4f}")
        # === Validation ===
        model.eval()
        metrics.reset()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)

                # Update metrics
                metrics.update(preds, labels)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)
    
        val_acc = val_correct / val_total
        val_loss /= val_total
        
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")
        
        val_metrics = metrics.compute()
        print(f"Val Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f} | "
          f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")
        
        train_stats['accuracy'].append(train_metrics['accuracy'].item())
        val_stats['accuracy'].append(val_metrics['accuracy'].item())
        train_stats['loss'].append(train_loss)
        val_stats['loss'].append(val_loss)
        train_stats['f1'].append(train_metrics['f1'].item())
        val_stats['f1'].append(val_metrics['f1'].item())
        train_stats['precision'].append(train_metrics['precision'].item())
        val_stats['precision'].append(val_metrics['precision'].item())
        train_stats['recall'].append(train_metrics['recall'].item())
        val_stats['recall'].append(val_metrics['recall'].item())
        train_stats['epoch'].append(epoch+1)
        val_stats['epoch'].append(epoch+1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_model_{model_name}_auto_threshed.pt")
    
        scheduler.step()
    
    print("Training complete. Best Val Acc:", best_val_acc)
    with open(f"best_model_{model_name}_auto_threshed_metrics.json", 'w') as f:
        f.write(json.dumps({"train_stats":train_stats, "val_stats":val_stats}))
    return model, layer_name

def load_trained_model(model_func, num_classes, path_to_model='./best_model.pt'):
    model, params_to_update, layer_name, model_name = model_func(num_classes)
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device("cpu")))
    model.eval()
    return model, layer_name

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os
import sys

sys.path.append(os.getcwd())

try:
    from main import CancerDataModule, CancerClassifier, setup_environment
except ImportError:
    sys.exit()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Prawdziwa klasa (Stan faktyczny)')
    plt.xlabel('Przewidziana klasa (Diagnoza AI)')
    plt.title('Macierz Pomylek - Wyniki Modelu')
    plt.show()

if __name__ == "__main__":
    device, BATCH_SIZE, IMG_SIZE = setup_environment()
    DATA_PATH = r"C:\Users\Kuba\Desktop\Projekt_ID374\data"
    
    print(" Ladowanie danych testowych")
    dm = CancerDataModule(DATA_PATH, BATCH_SIZE, IMG_SIZE)
    dm.setup()
    
    checkpoints_dir = "lightning_logs"
    
    if not os.path.exists(checkpoints_dir):
        print(f" Blad: Nie znaleziono folderu '{checkpoints_dir}'.")
        print("uruchom")
        sys.exit()

    try:
        versions = [d for d in os.listdir(checkpoints_dir) if "version_" in d]
        if not versions:
            raise FileNotFoundError("Brak folderow version_X w lightning_logs")
            
        latest_version = sorted(versions, key=lambda x: int(x.split('_')[1]))[-1]
        ckpt_path = os.path.join(checkpoints_dir, latest_version, "checkpoints")
        
        ckpt_files = os.listdir(ckpt_path)
        if not ckpt_files:
            raise FileNotFoundError("Folder checkpoints jest pusty!")
            
        ckpt_file = ckpt_files[0] 
        full_ckpt_path = os.path.join(ckpt_path, ckpt_file)
        
        print(f" Ladowanie modelu z pliku: {full_ckpt_path}")
        
        model = CancerClassifier.load_from_checkpoint(full_ckpt_path, num_classes=len(dm.test_ds.classes))
    except Exception as e:
        print(f" Blad przy szukaniu modelu: {e}")
        print("Sprawdz czy folder lightning_logs nie jest pusty.")
        sys.exit()

    model.eval()
    model.to(device)
    
    print("Generowanie diagnoz")
    
  
    all_preds = []
    all_labels = []
    
    test_loader = dm.test_dataloader()
    
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch 
            images = images.to(device)
            
        
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print(" Generowanie raportu...")
    class_names = dm.test_ds.classes
    
    print("\n" + "="*40)
    print("RAPORT KLASYFIKACJI")
    print("="*40)
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    print(" Wyswietlam macierz pomylek")
    plot_confusion_matrix(all_labels, all_preds, class_names)
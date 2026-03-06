import os
import shutil
import random
from pathlib import Path

SOURCE_DIR = r"C:\Users\Kuba\Downloads\cancer_histo\cancer_histo"

DEST_DIR = r"C:\Users\Kuba\Desktop\Projekt_ID374\data"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


def prepare_dataset():

    sources = {
        "lung_aca": os.path.join(SOURCE_DIR, "lung_image_sets", "lung_aca"),
        "lung_scc": os.path.join(SOURCE_DIR, "lung_image_sets", "lung_scc"),
        "lung_n":   os.path.join(SOURCE_DIR, "lung_image_sets", "lung_n"),
        "colon_aca": os.path.join(SOURCE_DIR, "colon_image_sets", "colon_aca"),
        "colon_n":   os.path.join(SOURCE_DIR, "colon_image_sets", "colon_n"),
    }


    if not os.path.exists(SOURCE_DIR):
        print(f"Blad: Nie znaleziono folderu zrodlowego: {SOURCE_DIR}")
        print("Sprawdz czy sciezka w SOURCE_DIR jest poprawna!")
        return

    print(f"Rozpoczynam przetwarzanie danych z: {SOURCE_DIR}")
    
    for class_name, class_path in sources.items():
        if not os.path.exists(class_path):
            print(f"Ostrzezenie: Nie znaleziono folderu klasy {class_name} w {class_path}")
            continue


        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images) 
        
        count = len(images)
        print(f"   Znaleziono {count} zdjec dla klasy: {class_name}")

        train_end = int(count * TRAIN_RATIO)
        val_end = train_end + int(count * VAL_RATIO)

 
        splits = {
            "train": images[:train_end],
            "val":   images[train_end:val_end],
            "test":  images[val_end:]
        }


        for split_name, split_images in splits.items():
            target_folder = os.path.join(DEST_DIR, split_name, class_name)
            os.makedirs(target_folder, exist_ok=True)

            for img_name in split_images:
                src_file = os.path.join(class_path, img_name)
                dst_file = os.path.join(target_folder, img_name)
                shutil.copy2(src_file, dst_file)
            
            print(f"      -> {split_name}: skopiowano {len(split_images)} plikow.")

    print("\nZAKONCZONO! Dane sa gotowe w folderze:", DEST_DIR)

if __name__ == "__main__":

    random.seed(42)
    prepare_dataset()
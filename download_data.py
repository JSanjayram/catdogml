import urllib.request
import zipfile
import os
import shutil
from pathlib import Path

def download_dataset():
    """Download Microsoft Cats and Dogs dataset"""
    url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
    zip_path = "kagglecatsanddogs_5340.zip"
    
    print("Downloading Microsoft Cats and Dogs dataset...")
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("temp_data")
    
    # Move and organize data
    source_path = Path("temp_data/PetImages")
    train_path = Path("data/train")
    
    # Clear existing data
    if train_path.exists():
        shutil.rmtree(train_path)
    train_path.mkdir(parents=True, exist_ok=True)
    
    # Create cat and dog folders
    (train_path / "cat").mkdir(exist_ok=True)
    (train_path / "dog").mkdir(exist_ok=True)
    
    # Copy 300 images each
    cat_source = source_path / "Cat"
    dog_source = source_path / "Dog"
    
    print("Organizing cat images...")
    cat_files = list(cat_source.glob("*.jpg"))[:300]
    for i, file in enumerate(cat_files):
        try:
            shutil.copy2(file, train_path / "cat" / f"cat_{i:03d}.jpg")
        except:
            continue
    
    print("Organizing dog images...")
    dog_files = list(dog_source.glob("*.jpg"))[:300]
    for i, file in enumerate(dog_files):
        try:
            shutil.copy2(file, train_path / "dog" / f"dog_{i:03d}.jpg")
        except:
            continue
    
    # Cleanup
    os.remove(zip_path)
    shutil.rmtree("temp_data")
    
    print(f"Dataset ready! Cat images: {len(list((train_path / 'cat').glob('*.jpg')))}")
    print(f"Dog images: {len(list((train_path / 'dog').glob('*.jpg')))}")

if __name__ == "__main__":
    download_dataset()
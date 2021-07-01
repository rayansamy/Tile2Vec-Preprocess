# Tile2Vec-Preprocess
Script to prepare the embedding part of Tile2Vec, the aim is to create with this anchor, neighbor and distant tiles from existing image folder.

# Prepraration :
```
pip install tqdm matplotlib numpy opencv-python
```

# Usage :
```
python -m transform --input_folder "<your-image-folder-path>" --output_folder "<folder-to-store-tiles>"
```
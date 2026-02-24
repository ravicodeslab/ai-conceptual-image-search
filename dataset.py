import os
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from similarity import cosine_similarity

class ImageDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.extractor = FeatureExtractor()

        if os.path.exists("features.npy") and os.path.exists("image_paths.npy"):
            self.features = np.load("features.npy")
            self.image_paths = np.load("image_paths.npy", allow_pickle=True)
        else:
            self._index_dataset()

    def _index_dataset(self):
        features = []
        image_paths = []

        for file in os.listdir(self.dataset_path):
            if file.lower().endswith(("jpg", "png", "jpeg")):
                path = os.path.join(self.dataset_path, file)
                image = Image.open(path).convert("RGB")
                feature = self.extractor.extract(image)
                features.append(feature)
                image_paths.append(path)

        self.features = np.array(features)
        self.image_paths = np.array(image_paths)

        np.save("features.npy", self.features)
        np.save("image_paths.npy", self.image_paths)

    def search(self, query_image, top_k=3):
        query_feature = self.extractor.extract(query_image)

        scores = []
        for feature, path in zip(self.features, self.image_paths):
            score = cosine_similarity(query_feature, feature)
            scores.append((score, path))

        scores.sort(reverse=True, key=lambda x: x[0])
        return scores[:top_k]
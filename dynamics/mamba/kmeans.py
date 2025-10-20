import numpy as np
from sklearn.cluster import KMeans as FullKMeans, MiniBatchKMeans

import torch


class KMeans:
    def __init__(self, encodings: np.ndarray, num_clusters: int):
        # normalize slot features so kmeans roughly clusters based on cosine similarity
        encodings = encodings / np.linalg.norm(encodings, axis=-1, keepdims=True)
        self.encodings = encodings
        self.num_clusters = num_clusters
        self.kmeans_init = False

    def init_kmeans(self):
        self.kmeans_init = True
        encodings = self.encodings
        num_clusters = self.num_clusters

        if encodings.shape[-1] <= 256:
            kmeans = FullKMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
        else:
            kmeans = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', random_state=42, batch_size=1024)
        kmeans.fit(encodings)

        cluster_centers = kmeans.cluster_centers_                                       # (num_clusters, slot_dim)
        cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=-1, keepdims=True)
        cos_sim = (cluster_centers[None] * encodings[:, None]).sum(-1)                  # (num_samples, num_clusters)
        average_cos_sim = cos_sim.max(-1).mean()
        print(f"KMeans fitted {len(encodings)} slots for {num_clusters} clusters: "
              f"average cosine similarity = {average_cos_sim:.3f}")

        labels = cos_sim.argmax(-1)
        sorted_indices = np.argsort(labels)  # Sort indices by cluster labels
        labels = labels[sorted_indices]
        encodings = encodings[sorted_indices]

        cluster_counts = np.bincount(labels, minlength=num_clusters)
        cluster_starts = np.zeros(num_clusters, dtype=int)
        cluster_starts[1:] = np.cumsum(cluster_counts[:-1])

        self.encodings = encodings
        self.cluster_centers = cluster_centers
        self.cluster_counts = cluster_counts
        self.cluster_starts = cluster_starts

    def sample(self, shape):
        if isinstance(shape, int):
            shape = (shape,)
        num_samples = np.prod(shape)
        label = np.random.randint(0, len(self.encodings), size=num_samples)
        sampled_slots = self.encodings[label].reshape(shape + self.encodings.shape[-1:])
        return sampled_slots

    def sample_from_same_cluster(self, slots):
        # slots: (n_samples, n_features)
        if isinstance(slots, torch.Tensor):
            slots = slots.detach().cpu().numpy()

        if not self.kmeans_init:
            self.init_kmeans()

        label = np.argmax(np.dot(slots, self.cluster_centers.T), axis=-1)           # (n_samples,)
        sample_offsets = np.random.randint(0, self.cluster_counts[label])       # (n_samples,)
        sample_indices = self.cluster_starts[label] + sample_offsets
        sampled_slots = self.encodings[sample_indices]
        return sampled_slots

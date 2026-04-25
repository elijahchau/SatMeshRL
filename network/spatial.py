"""Spatial indexing helpers using a KD-tree.

The `SpatialIndex` wraps `scipy.spatial.cKDTree` and provides a
convenience `k_nearest` method that returns k nearest neighbors for
each point in the index (excluding the point itself).
"""

import numpy as np
from scipy.spatial import cKDTree


class SpatialIndex:
    """KD-tree backed spatial index for satellite positions.

    Parameters
    - positions: mapping id -> (x,y,z)
    """

    def __init__(self, positions):
        self.ids = list(positions.keys())
        # coords shape: (N, 3)
        self.coords = np.array(list(positions.values()))
        # Build the KD-tree once for repeated neighbor queries.
        self.tree = cKDTree(self.coords)

    def k_nearest(self, k):
        """
        Return distances and indices of k nearest neighbors for every point.

        The method queries `k+1` neighbors and strips the first result
        (the point itself) to return exactly `k` neighbors per node.
        """
        dists, idxs = self.tree.query(self.coords, k=k + 1)
        return dists[:, 1:], idxs[:, 1:]

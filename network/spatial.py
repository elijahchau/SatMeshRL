"""Spatial indexing helpers using a KD-tree.

The `SpatialIndex` wraps `scipy.spatial.cKDTree` and provides
vectorized nearest-neighbor queries used during graph construction.
"""

import numpy as np
from scipy.spatial import cKDTree


class SpatialIndex:
    """KD-tree backed spatial index for satellite positions.

    Inputs:
    - positions: mapping node id to 3D position in kilometers

    The index stores the node ids and a dense coordinate array so
    nearest-neighbor queries can return both distances and indices.
    """

    def __init__(self, positions):
        self.ids = list(positions.keys())
        if not self.ids:
            raise ValueError("SpatialIndex requires at least one position.")

        coords = np.array(list(positions.values()), dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("Positions must be a mapping of id -> (x, y, z).")

        # coords shape: (N, 3)
        self.coords = coords
        # Build the KD-tree once for repeated neighbor queries.
        self.tree = cKDTree(self.coords)

    def radius_neighbors(self, max_dist):
        """
        Return neighbor indices and distances for all points within `max_dist`.

        Output:
        - list_of_indices: list where element i is a list of neighbor indices
          within `max_dist` (excluding the point itself)
        - list_of_dists: list where element i is a numpy array of distances
          corresponding to the indices in list_of_indices[i]

        This method is suitable when building graphs using a fixed radius
        threshold instead of a fixed k-nearest neighbor count.
        """

        # query_ball_point returns lists of indices (including the point itself)
        raw = self.tree.query_ball_point(self.coords, r=max_dist)
        list_of_indices = []
        list_of_dists = []

        for i, idxs in enumerate(raw):
            # remove self index if present
            neigh = [j for j in idxs if j != i]
            if not neigh:
                list_of_indices.append([])
                list_of_dists.append(np.array([]))
                continue

            nbr_coords = self.coords[neigh]
            dists = np.linalg.norm(nbr_coords - self.coords[i], axis=1)
            list_of_indices.append(neigh)
            list_of_dists.append(dists)

        return list_of_indices, list_of_dists

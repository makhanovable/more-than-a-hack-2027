import numpy as np

MISSING_BIN = 0


class Binner:
    def __init__(self, max_bins=32):
        self.max_bins = max_bins
        self.edges = None

    def fit(self, features):
        quantiles = np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1]
        self.edges = []
        for column in features.T:
            finite = column[np.isfinite(column)]
            if len(finite) == 0:
                self.edges.append(np.array([]))
                continue
            self.edges.append(np.unique(np.quantile(finite, quantiles)))
        return self

    def transform(self, features):
        out = np.zeros(features.shape, dtype=np.uint8)
        for index, column in enumerate(features.T):
            finite = np.isfinite(column)
            out[finite, index] = 1 + np.searchsorted(self.edges[index], column[finite], side="right")
        return out

    @property
    def n_bins(self):
        return self.max_bins + 1


class Booster:
    def __init__(self, n_trees=150, depth=3, learning_rate=0.1, l2=1.0, min_child_weight=20.0,
                 column_fraction=0.5, row_fraction=0.8, n_bins=33, seed=0):
        self.n_trees = n_trees
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2 = l2
        self.min_child_weight = min_child_weight
        self.column_fraction = column_fraction
        self.row_fraction = row_fraction
        self.n_bins = n_bins
        self.seed = seed
        self.base = 0.0
        self.trees = []

    def fit(self, bins, target, weight=None):
        n, n_features = bins.shape
        weight = np.ones(n) if weight is None else np.asarray(weight, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        rng = np.random.RandomState(self.seed)

        positive = (weight * target).sum() / weight.sum()
        positive = min(max(positive, 1e-6), 1.0 - 1e-6)
        self.base = float(np.log(positive / (1.0 - positive)))
        raw = np.full(n, self.base)
        self.trees = []

        n_columns = max(1, int(round(self.column_fraction * n_features)))
        for _ in range(self.n_trees):
            probability = 1.0 / (1.0 + np.exp(-raw))
            gradient = weight * (probability - target)
            hessian = weight * np.maximum(probability * (1.0 - probability), 1e-6)

            rows = np.flatnonzero(rng.rand(n) < self.row_fraction) \
                if self.row_fraction < 1.0 else np.arange(n)
            columns = np.sort(rng.choice(n_features, n_columns, replace=False))
            tree = self._grow(bins[rows], gradient[rows], hessian[rows], columns)
            self.trees.append(tree)
            raw += self._apply(tree, bins)
        return self

    def _grow(self, bins, gradient, hessian, columns):
        n = len(gradient)
        node = np.zeros(n, dtype=np.int64)
        features, thresholds = [], []
        for level in range(self.depth):
            n_nodes = 1 << level
            total_g = np.bincount(node, weights=gradient, minlength=n_nodes)
            total_h = np.bincount(node, weights=hessian, minlength=n_nodes)
            parent = total_g ** 2 / (total_h + self.l2)

            best_gain = np.zeros(n_nodes)
            best_feature = np.zeros(n_nodes, dtype=np.int64)
            best_threshold = np.full(n_nodes, self.n_bins, dtype=np.int64)
            base_key = node * self.n_bins
            for feature in columns:
                key = base_key + bins[:, feature]
                size = n_nodes * self.n_bins
                hist_g = np.bincount(key, weights=gradient, minlength=size).reshape(n_nodes, -1)
                hist_h = np.bincount(key, weights=hessian, minlength=size).reshape(n_nodes, -1)
                left_g = np.cumsum(hist_g, axis=1)[:, :-1]
                left_h = np.cumsum(hist_h, axis=1)[:, :-1]
                right_g = total_g[:, None] - left_g
                right_h = total_h[:, None] - left_h
                gain = left_g ** 2 / (left_h + self.l2) + right_g ** 2 / (right_h + self.l2) \
                    - parent[:, None]
                gain[(left_h < self.min_child_weight) | (right_h < self.min_child_weight)] = 0.0
                candidate = gain.argmax(axis=1)
                candidate_gain = gain[np.arange(n_nodes), candidate]
                better = candidate_gain > best_gain
                best_gain[better] = candidate_gain[better]
                best_feature[better] = feature
                best_threshold[better] = candidate[better]
            features.append(best_feature)
            thresholds.append(best_threshold)
            go_right = bins[np.arange(n), best_feature[node]] > best_threshold[node]
            node = 2 * node + go_right

        n_leaves = 1 << self.depth
        leaf_g = np.bincount(node, weights=gradient, minlength=n_leaves)
        leaf_h = np.bincount(node, weights=hessian, minlength=n_leaves)
        values = -self.learning_rate * leaf_g / (leaf_h + self.l2)
        return features, thresholds, values

    @staticmethod
    def _apply(tree, bins):
        features, thresholds, values = tree
        rows = np.arange(len(bins))
        node = np.zeros(len(bins), dtype=np.int64)
        for best_feature, best_threshold in zip(features, thresholds):
            node = 2 * node + (bins[rows, best_feature[node]] > best_threshold[node])
        return values[node]

    def decision_function(self, bins):
        raw = np.full(len(bins), self.base)
        for tree in self.trees:
            raw += self._apply(tree, bins)
        return raw

    def feature_usage(self, n_features):
        usage = np.zeros(n_features, dtype=np.int64)
        for features, thresholds, _ in self.trees:
            for best_feature, best_threshold in zip(features, thresholds):
                real = best_threshold < self.n_bins
                np.add.at(usage, best_feature[real], 1)
        return usage

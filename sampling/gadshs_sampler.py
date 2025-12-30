import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch
import networkx as nx
import community as community_louvain

class GADHSSampler:
    def __init__(self, k=5, alpha=1.0, beta=0.5, gat_hidden=32, gat_heads=4, epochs=50, device='cpu'):
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.gat_hidden = gat_hidden
        self.gat_heads = gat_heads
        self.epochs = epochs
        self.device = device

    def fit_resample(self, X, y):
        # X: ndarray shape (n_samples, n_features), y: labels
        X = np.asarray(X)
        y = np.asarray(y)
        minority_idx = np.where(y == 1)[0]
        majority_idx = np.where(y == 0)[0]

        # 1. Build kNN graph
        nbrs = NearestNeighbors(n_neighbors=self.k).fit(X)
        distances, indices = nbrs.kneighbors(X)
        edge_index = []
        for i, neighs in enumerate(indices):
            for j in neighs:
                edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # 2. Density estimation for minority
        kde = KernelDensity(kernel='gaussian').fit(X[minority_idx])
        densities = np.exp(kde.score_samples(X[minority_idx]))
        # Normalize density
        inv_density = 1.0 / (densities + 1e-6)
        weights = inv_density / inv_density.sum()

        # 3. Prepare graph data for GAT
        x_feat = torch.tensor(X, dtype=torch.float)
        data = Data(x=x_feat, edge_index=edge_index)
        data = data.to(self.device)

        # GAT model
        class GAT(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, heads):
                super().__init__()
                self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
                self.conv2 = GATConv(hidden_channels*heads, 1, heads=1)

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = torch.relu(x)
                x = self.conv2(x, edge_index)
                return x.squeeze()

        model = GAT(X.shape[1], self.gat_hidden, self.gat_heads).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            # self-supervised: reconstruct neighbor labels? using edges
            loss = ((out - out.mean())**2).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            attention_scores = model(data.x, data.edge_index).cpu().numpy()

        # 4. Oversampling minority
        synthetic_samples = []
        for idx, v in enumerate(minority_idx):
            n_new = int(self.alpha * weights[idx] * len(minority_idx))
            neighs = indices[v]
            scores = attention_scores[neighs]
            probs = np.exp(scores) / np.sum(np.exp(scores))
            for _ in range(n_new):
                j = np.random.choice(neighs, p=probs)
                lam = np.random.rand()
                new = X[v] + lam * (X[j] - X[v])
                synthetic_samples.append(new)
        X_syn = np.array(synthetic_samples)
        y_syn = np.ones(len(X_syn), dtype=int)

        # 5. Undersampling majority via community detection
        G = nx.Graph()
        for u, v in edge_index.t().tolist():
            G.add_edge(u, v)
        partition = community_louvain.best_partition(G)
        # find boundary communities
        maj_communities = [partition[i] for i in majority_idx]
        # sample removal
        remove_idx = []
        for idx in majority_idx:
            if np.random.rand() < self.beta:
                remove_idx.append(idx)
        keep_idx = [i for i in majority_idx if i not in remove_idx]

        # final dataset
        X_res = np.vstack([X[keep_idx], X[minority_idx], X_syn])
        y_res = np.hstack([np.zeros(len(keep_idx)), np.ones(len(minority_idx)), y_syn])
        return X_res, y_res

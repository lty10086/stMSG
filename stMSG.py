import torch
import time
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
import anndata as ad
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, LocalOutlierFactor


class AutoEncoder(nn.Module):

    def __init__(self, input_dim: int, latent_size: int):
        super(AutoEncoder, self).__init__()
        
        hidden_size = input_dim // 2
        
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, latent_size)
        self.layer_norm2 = nn.LayerNorm(latent_size)
        
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_dim)

    def forward(self, x):
        encode = self.fc1(x)
        encode = self.layer_norm1(encode)
        latent = self.fc2(encode)
        latent = self.layer_norm2(latent)
        
        decode = self.fc3(latent)
        reconstructed = self.fc4(decode)
        
        return latent, reconstructed

def pairwise_sqdist(a, b):
    a_norm = (a ** 2).sum(dim=-1).unsqueeze(-1)
    b_norm = (b ** 2).sum(dim=-1).unsqueeze(-2)
    ab = torch.matmul(a, b.transpose(-2, -1))
    return a_norm + b_norm - 2 * ab

def mmd_rbf(x, y, sigmas=(1, 2, 4, 8, 16, 32, 64, 128, 256)):
    device, dtype = x.device, x.dtype

    if x.size(0) > 1e4:
        center = x.mean(dim=0)
        dists = torch.sum((x - center) ** 2, dim=1)
        _, indices = torch.topk(dists, max(y.size(0),5000), largest=False, sorted=False)
        x = x[indices]
    else:
        x = x
        
    sigmas = torch.tensor(sigmas, dtype=dtype, device=device)
    beta = (1.0 / (2.0 * sigmas ** 2)).view(-1, 1, 1)

    d_xx = pairwise_sqdist(x, x).unsqueeze(0)
    d_yy = pairwise_sqdist(y, y).unsqueeze(0)
    d_xy = pairwise_sqdist(x, y).unsqueeze(0)

    k_xx = torch.exp(-beta * d_xx).mean(dim=0)
    k_yy = torch.exp(-beta * d_yy).mean(dim=0)
    k_xy = torch.exp(-beta * d_xy).mean(dim=0)

    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    
def train_autoencoder(adata_sc, adata_st, common_genes, latent_size, a, lr, k, epochs, device, map_view):

    criterion = nn.MSELoss()
    
    print('ST Cell autoencoder begins training...')
    sc_norm_data = adata_sc[:,common_genes].X.copy()
    X_sc_tensor = torch.tensor(sc_norm_data.copy(), dtype=torch.float).to(device)
    st_norm_data = adata_st[:,common_genes].X.copy()
    X_st_tensor = torch.tensor(st_norm_data.copy(), dtype=torch.float).to(device)
    
    modelST = AutoEncoder(X_st_tensor.shape[1], latent_size).to(device)
    optimizerST = torch.optim.Adam(modelST.parameters(), lr=lr)
    for epoch in range(epochs):
        modelST.train()
        optimizerST.zero_grad()
        z_st, xhat_st = modelST(X_st_tensor)
        z_sc, xhat_sc = modelST(X_sc_tensor)
        loss_rec = criterion(xhat_st, X_st_tensor)
        loss_mmd_z = mmd_rbf(z_sc, z_st)
        loss = (1-a)*loss_rec + a*loss_mmd_z
        loss.backward()
        optimizerST.step()
        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {loss.item():.4f} (RecST: {loss_rec.item():.4f}, MMDz: {loss_mmd_z.item():.4f})")

    modelST.eval()
    with torch.no_grad():
        st_embedding, _ = modelST(X_st_tensor)
        sc_embedding, _ = modelST(X_sc_tensor)
        
    sc_embedding_np = sc_embedding.detach().cpu().numpy()
    st_embedding_np = st_embedding.detach().cpu().numpy()

    if map_view:
        adata_joint = ad.concat([adata_sc, adata_st], axis=0, label="source", keys=["sc", "st"], index_unique=None)
        X_latent = np.vstack([sc_embedding_np, st_embedding_np])
        adata_joint.obsm["X_latent"] = X_latent
        sc.pp.neighbors(adata_joint, n_neighbors=15, use_rep="X_latent", random_state=0)
        sc.tl.umap(adata_joint, random_state=0)
        sc.pl.umap(adata_joint, color=["source"], size=10, frameon=False)

    nbrs = NearestNeighbors(n_neighbors=k).fit(sc_embedding_np)
    distances, indices = nbrs.kneighbors(st_embedding_np)
    sc_all_data = adata_sc.X.copy()
    Xsc = np.zeros((st_norm_data.shape[0], sc_all_data.shape[1]))
    for i in range(st_embedding_np.shape[0]):
        nearest_cells = sc_all_data[indices[i]]
        Xsc[i] = np.mean(nearest_cells, axis=0)
    
    adata_st_map = ad.AnnData(X=Xsc)
    adata_st_map.obs_names = adata_st.obs_names.copy()
    adata_st_map.var_names = adata_sc.var_names.copy()
    adata_st_map.obsm['spatial'] = adata_st.obsm['spatial']
    adata_st_map.obsm['X_latent'] = st_embedding_np
    
    for gene in common_genes:
        adata_st_map[:, gene].X = adata_st[:, gene].X.copy()
    return adata_st_map

def Pre_clustering(adata, k, view):

    print('Preliminary clustering...')
    adata_st_pre = adata.copy()
    n_components = min(adata_st_pre.n_obs, adata_st_pre.n_vars) - 1
    n_components = min(40, n_components)
    sc.pp.pca(adata_st_pre, n_comps=n_components)
    sc.pp.neighbors(adata_st_pre, n_neighbors=k, n_pcs=n_components)
    sc.tl.leiden(adata_st_pre, resolution=1.0)
    sc.tl.umap(adata_st_pre)
    if view:
        sc.pl.umap(adata_st_pre, color='leiden', size=30)
    cluster_labels = adata_st_pre.obs['leiden'].astype('category').cat.codes.values
    return adata_st_pre.obsm['X_umap'], cluster_labels
    
def Create_cell_network(adata_st, adata_st_map, k):

    Z_umap, cluster_labels = Pre_clustering(adata_st, k, None)
    Z_umap_map, _ = Pre_clustering(adata_st_map, k, None)

    print('Searching for scattered cells...')
    spatial_coordinates = adata_st.obsm['spatial']
    
    unique_clusters = np.unique(cluster_labels)
    outlier_list = []

    for cluster in unique_clusters:
        cluster_cells = np.where(cluster_labels == cluster)[0]
        cluster_data = Z_umap_map[cluster_cells]

        lof = LocalOutlierFactor(n_neighbors=len(cluster_cells)//3, contamination=0.3)
        y_pred = lof.fit_predict(cluster_data)

        outliers = cluster_data[y_pred == -1]
        inliers = cluster_data[y_pred == 1]

        '''
        plt.figure(figsize=(8, 6))
        plt.scatter(inliers[:, 0], inliers[:, 1], color='blue', label=f'Cluster {cluster} Normal', alpha=0.6)
        plt.scatter(outliers[:, 0], outliers[:, 1], color='red', label=f'Cluster {cluster} Outliers', alpha=0.6)
        plt.title(f"LOF Outlier Detection for Cluster {cluster}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend(loc='best')
        plt.show()
        '''
        
        outlier_indices_local = np.where(y_pred == -1)[0]
        outlier_indices_global = cluster_cells[outlier_indices_local]
        outlier_list.append(outlier_indices_global)
    outlier_list = np.concatenate(outlier_list)
    
    print('Creating cell network...')
    nn = NearestNeighbors(n_neighbors=min(k, Z_umap.shape[0]))
    nn.fit(Z_umap)
    _, indices = nn.kneighbors(Z_umap)

    G1 = np.zeros((Z_umap.shape[0], Z_umap.shape[0]))
    
    for i in range(Z_umap.shape[0]):
        for neigh in indices[i]:
            if i < neigh and cluster_labels[i] == cluster_labels[neigh]:
                G1[i, neigh] = 1
                G1[neigh, i] = 1

    G2 = G1.copy()
    nn = NearestNeighbors(n_neighbors=min(k, spatial_coordinates.shape[0]))
    nn.fit(spatial_coordinates)
    _, indices = nn.kneighbors(spatial_coordinates)
    for i in outlier_list:
        neigh = indices[i]
        neigh = neigh[neigh != i]
        same = neigh[cluster_labels[neigh] == cluster_labels[i]]
        if same.size > 0:
            for j in same:
                G2[i, j] = 1
                G2[j, i] = 1
    return G1,G2

class GraphAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_size: int):
        super(GraphAutoEncoder, self).__init__()
        hidden_size = input_dim // 2
        self.gc1 = GCNConv(input_dim, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.gc2 = GCNConv(hidden_size, latent_size)
        self.layer_norm2 = nn.LayerNorm(latent_size)
        
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_dim)

    def forward(self, x, edge_index):
        hidden = F.relu(self.gc1(x, edge_index))
        hidden = self.layer_norm1(hidden)
        latent = F.relu(self.gc2(hidden, edge_index))
        latent = self.layer_norm2(latent)
        decode = F.relu(self.fc1(latent))
        reconstructed = self.fc2(decode)
        return latent, reconstructed
    
def train_graphAutoEncoder(adata_st_map, G, latent_size, lr, epochs, device):
    print('Cell graphautoencoder(all genes) begins training......')
    edge_index, _ = dense_to_sparse(torch.tensor(G, dtype=torch.float, device=device))
    data = Data(x=torch.tensor(adata_st_map.X, dtype=torch.float, device=device), edge_index=edge_index)
    model = GraphAutoEncoder(input_dim=adata_st_map.X.shape[1], latent_size=latent_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    latent_map = torch.tensor(adata_st_map.obsm['X_latent'].copy(), dtype=torch.float).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        latent, reconstructed = model(data.x, data.edge_index)
        loss_rec = criterion(reconstructed, data.x)
        loss_mmd_z = criterion(latent, latent_map)
        loss = loss_rec + loss_mmd_z 

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {loss.item():.4f} (RecST: {loss_rec.item():.4f}, Recz: {loss_mmd_z.item():.4f})")
    
    model.eval()
    with torch.no_grad():
        _, imp = model(data.x, data.edge_index)
    Imp = imp.detach().cpu().numpy()
    Imp[Imp<0]= -Imp[Imp<0]
    return Imp
# Embeddings and Outlier

# Imports

import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import warnings
warnings.filterwarnings('ignore')
from config import DATA_DIR, OUTPUT_DIR

"""
The embeddings created by GAT and GT may vary from run to run. This is because:

The models use random initialization of weights.
There are stochastic operations, such as dropout, used during training.
The optimizer may follow different paths due to data order or minor numerical differences.
To ensure reproducibility, set the seeds for the random generators of PyTorch, NumPy, and Python and turn off dropout during inference. 'Dropout' is a technique used in neural networks to prevent overfitting. It randomly sets a fraction of the input units to 0 at each update during training. Even then, slight variations may occur due to parallel operations or hardware differences.

The set_seed block below is designed to ensure the reproducibility of experiments across all code snippets that use randomization in Python, NumPy, and PyTorch (including CUDA and MPS), providing you with reassurance in your experiments.

This means that:
 - The initial weights of the models (GAT, GT, autoencoders, etc.) will always be the same from run to run.
 - Random operations (such as shuffling and parameter initialization) will yield the same result.
 - Training results will be the same as long as there are no other sources of randomness outside the control of these libraries, giving you a sense of control and empowerment in your experiments.

The seed is set before any model creation or random function call.
It covers the three primary sources of randomness: Python (random), NumPy, and PyTorch (including CUDA and MPS), providing a comprehensive and secure approach to reproducibility.
The PyTorch backend determinism is enabled to ensure that operations are reproducible.
Limitations:

Some GPU operations may still exhibit slight variations depending on the hardware and driver.
If you use other libraries that generate randomness (e.g., TensorFlow, sci-kit-learn in parallelism), you may need to set seeds for them as well.

Summary: With this block, the code will be reproducible in any part that depends on randomness from Python, NumPy, or PyTorch, both on CPU and CUDA/MPS.
"""

def set_seed(seed):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# Utility class to represent large graph components

class GraphComponent:
    """
    Represents a graph component with its NetworkX graph, network id, node and edge counts.
    """
    def __init__(self, network_id, nx_graph):
        # Store the unique identifier for the network/component
        self.network_id = network_id
        # Store the NetworkX graph object representing the component
        self.nx_graph = nx_graph
        # Store the number of nodes in the component
        self.num_nodes = nx_graph.number_of_nodes()
        # Store the number of edges in the component
        self.num_edges = nx_graph.number_of_edges()

    def __repr__(self):
        # Custom string representation for easy inspection
        return (f"GraphComponent(network_id={self.network_id}, "
                f"nodes={self.num_nodes}, edges={self.num_edges})")

def build_graph_components(large_components_df, nx_full_graph):
    """
    Builds a list of GraphComponent objects for each large component.

    Args:
        large_components_df (pd.DataFrame): DataFrame with 'network' column, listing selected large components.
        nx_full_graph (networkx.Graph): The full graph containing all nodes and edges.

    Returns:
        List[GraphComponent]: List of GraphComponent objects, one for each large component.
    """
    components_list = []
    # Get all connected components as a list for indexing by network id
    all_components = list(nx.connected_components(nx_full_graph))
    for _, row in large_components_df.iterrows():
        network_id = int(row['network'])
        # Components are indexed from 1 in the DataFrame, so subtract 1 for zero-based index
        component_nodes = all_components[network_id - 1]
        # Extract the subgraph corresponding to the current component
        subgraph = nx_full_graph.subgraph(component_nodes).copy()
        # Create a GraphComponent object and add it to the list
        components_list.append(GraphComponent(network_id, subgraph))
    return components_list

# Define the GAT (Graph Attention Network) model for node embedding generation
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(GAT, self).__init__()
        # First GAT layer: input features to 8*8 hidden features (8 heads)
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        # Second GAT layer: 8*8 hidden features to out_channels (1 head, concatenated)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=True, dropout=0.6)
        # Decoder: linear layer to reconstruct the input from embeddings
        self.decoder = torch.nn.Linear(out_channels, num_nodes)

    def forward(self, x, edge_index):
        # Apply dropout and first GAT layer with ELU activation
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        # Apply dropout and second GAT layer
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def reconstruct(self, x, edge_index):
        # Forward pass and decode to reconstruct the input features
        z = self.forward(x, edge_index)
        return self.decoder(z)
    
# Define the Graph Transformer model for node embedding generation
class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(GraphTransformer, self).__init__()
        # First TransformerConv layer: input features to 8*8 hidden features (8 heads)
        self.conv1 = TransformerConv(in_channels, 8, heads=8, dropout=0.6)
        # Second TransformerConv layer: 8*8 hidden features to out_channels (1 head, concatenated)
        self.conv2 = TransformerConv(8 * 8, out_channels, heads=1, concat=True, dropout=0.6)
        # Decoder: linear layer to reconstruct the input from embeddings
        self.decoder = torch.nn.Linear(out_channels, num_nodes)

    def forward(self, x, edge_index):
        # Apply dropout and first TransformerConv layer with ELU activation
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        # Apply dropout and second TransformerConv layer
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def reconstruct(self, x, edge_index):
        # Forward pass and decode to reconstruct the input features
        z = self.forward(x, edge_index)
        return self.decoder(z)
    
# Outlier Detection

# Function to compute Euclidean distance matrix
def compute_euclidian_distance_matrix(X):
    S = distance.cdist(X, X, 'euclidean')
    return S

# Function to compute Jensen-Shannon distance matrix
def compute_jensen_distance_matrix(X):
    S = distance.cdist(X, X, 'jensenshannon')
    return S

# Function to compute probability distributions from distance matrix
def probabilidade(mdisteuc):
    mdistnew = [[x for idxx, x in enumerate(X) if idxx != idx] for idx, X in enumerate(mdisteuc)]
    mdistnew = np.array(mdistnew)
    max_d = mdistnew.max(axis=1)
    min_d = mdistnew.min(axis=1)
    N = len(mdistnew)
    k = (N-1)//10 if (N-1)//10 > 0 else 1
    P = np.zeros((N, k), dtype=np.float64)
    for i in range(N):
        max_i = max_d[i]
        min_i = min_d[i]
        d = mdistnew[i, :]
        bins = np.linspace(min_i, max_i, k+1)
        hist, _ = np.histogram(d, bins=bins)
        prob = hist / (N-1)
        P[i] = prob
    return P

# Function to get the available device (cuda, mps, or cpu)
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

# Function to compute cosine similarity matrix
def compute_cosine_similarity_matrix(X):
    S = cosine_similarity(X)
    return S

# Function to compute autocorrelations between all pairs of rows in X
def compute_autocorrelations(X):
    N = X.shape[0]
    S = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            S[i, j] = np.correlate(X[i], X[j], mode='valid')[0]
    return S

# Function to compute mean similarity for each row
def compute_similarity(S):
    mean_similarities = np.mean(S, axis=1)
    return mean_similarities

# Autoencoder neural network class
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )
    # Forward pass for autoencoder
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Function to train the autoencoder and return reconstruction error
def train_autoencoder(X, epochs, batch_size=16, learning_rate=0.001):
    device_used = get_device()
    model = Autoencoder(X.shape[1]).to(device_used)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device_used)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, X_tensor)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        reconstructed = model(X_tensor).cpu().numpy()
    reconstruction_error = np.mean(np.abs(X - reconstructed), axis=1)
    return reconstruction_error

# Outlier detection class with multiple methods
class OutlierDetector:
    def __init__(self, contamination=0.164):
        self.contamination = contamination

    # Apply Isolation Forest method
    def apply_isolation_forest(self, X_normalized):
        isf = IsolationForest(
            n_estimators=100,
            max_samples=len(X_normalized),
            contamination=self.contamination,
            random_state=42
        )
        isf.fit(X_normalized)
        scores = isf.decision_function(X_normalized)
        return scores

    # Apply Elliptic Envelope method
    def apply_elliptic_envelope(self, X_normalized):
        cov = EllipticEnvelope(
            contamination=self.contamination,
            random_state=42
        )
        cov.fit(X_normalized)
        scores = cov.decision_function(X_normalized)
        return scores

    # Apply SGD One-Class SVM method
    def apply_sgd_one_class_svm(self, X_normalized):
        svm = SGDOneClassSVM(
            nu=self.contamination,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            shuffle=True,
            verbose=0,
            random_state=42,
            learning_rate='optimal',
            eta0=0.0,
            power_t=0.5,
            warm_start=False,
            average=False
        )
        svm.fit(X_normalized)
        scores = svm.decision_function(X_normalized)
        return scores

    # Calculate OS1 and OS2 outlier scores
    def calc_os1_os2(self, X_normalized):
        mdisteuc = compute_euclidian_distance_matrix(X_normalized)
        mdisteucnorm = (mdisteuc - mdisteuc.min()) / (mdisteuc.max() - mdisteuc.min() + 1e-8)  # Normalize to [0, 1]    
        mdisteucnorm = np.clip(mdisteucnorm, 0, 1)  # Ensure values are in [0, 1]
        P = probabilidade(mdisteucnorm)
        mdistjen = compute_jensen_distance_matrix(P)
        mdistjennorm = (mdistjen - mdistjen.min()) / (mdistjen.max() - mdistjen.min() + 1e-8)
        mdistjennorm = np.clip(mdistjen, 0, 1)  # Ensure values are in [0, 1]
        OS1_probs = mdisteucnorm.sum(axis=1)
        OS1_probs = (OS1_probs - OS1_probs.min()) / (OS1_probs.max() - OS1_probs.min() + 1e-8)  # Normalize to [0, 1]
        OS1_probs = np.clip(OS1_probs, 0, 1) # Ensure values are in [0, 1]
        OS2_probs = mdistjennorm.sum(axis=1)
        OS2_probs = (OS2_probs - OS2_probs.min()) / (OS2_probs.max() - OS2_probs.min() + 1e-8)  # Normalize to [0, 1]
        OS2_probs = np.clip(OS2_probs, 0, 1)  # Ensure values are in [0, 1]
        return OS1_probs, OS2_probs

    # Calculate CoADA outlier scores using cosine similarity and autoencoder
    def calc_coada(self, X, epochs=150):
        C = compute_cosine_similarity_matrix(X)
        similaritiesCo = compute_similarity(C).reshape(-1, 1)
        CoADA = train_autoencoder(similaritiesCo, epochs=epochs)
        CoADA = CoADA / CoADA.max() if CoADA.max() != 0 else CoADA
        CoADA = (CoADA - CoADA.min()) / (CoADA.max() - CoADA.min() + 1e-8)  # Normalize to [0, 1]
        CoADA = np.clip(CoADA, 0, 1)    # Ensure values are in [0, 1]      
        return CoADA

    # Deep SVDD method
    def calc_svdd(self, X, epochs=100, rep_dim=8, lr=1e-3):
        class SVDDNet(nn.Module):
            def __init__(self, input_dim, rep_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, rep_dim)
                )
            def forward(self, x):
                return self.net(x)
        device = get_device()
        model = SVDDNet(X.shape[1], rep_dim=rep_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            c = model(X_tensor).mean(dim=0)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = torch.mean(torch.sum((outputs - c) ** 2, dim=1))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            dist = torch.sum((outputs - c) ** 2, dim=1).cpu().numpy()
        dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
        dist = np.clip(dist, 0, 1)  # Ensure values are in [0, 1]
        return dist

    # # AnoGAN method
    # def calc_anogan(self, X, latent_dim=16, epochs=200, lr=0.0002):
    #     class Generator(nn.Module):
    #         def __init__(self, latent_dim, data_dim):
    #             super().__init__()
    #             self.model = nn.Sequential(
    #                 nn.Linear(latent_dim, 64),
    #                 nn.ReLU(),
    #                 nn.Linear(64, data_dim)
    #             )
    #         def forward(self, z):
    #             return self.model(z)
    #     class Discriminator(nn.Module):
    #         def __init__(self, data_dim):
    #             super().__init__()
    #             self.model = nn.Sequential(
    #                 nn.Linear(data_dim, 64),
    #                 nn.ReLU(),
    #                 nn.Linear(64, 1),
    #                 nn.Sigmoid()
    #             )
    #         def forward(self, x):
    #             return self.model(x)
    #     device = get_device()
    #     X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    #     data_dim = X.shape[1]
    #     G = Generator(latent_dim, data_dim).to(device)
    #     D = Discriminator(data_dim).to(device)
    #     optimizer_G = optim.Adam(G.parameters(), lr=lr)
    #     optimizer_D = optim.Adam(D.parameters(), lr=lr)
    #     criterion = nn.BCELoss()
    #     batch_size = min(32, X.shape[0])
    #     for epoch in range(epochs):
    #         idx = np.random.randint(0, X.shape[0], batch_size)
    #         real = X_tensor[idx]
    #         valid = torch.ones((batch_size, 1), device=device)
    #         fake = torch.zeros((batch_size, 1), device=device)
    #         z = torch.randn((batch_size, latent_dim), device=device)
    #         gen_data = G(z)
    #         optimizer_D.zero_grad()
    #         loss_real = criterion(D(real), valid)
    #         loss_fake = criterion(D(gen_data.detach()), fake)
    #         loss_D = (loss_real + loss_fake) / 2
    #         loss_D.backward()
    #         optimizer_D.step()
    #         optimizer_G.zero_grad()
    #         loss_G = criterion(D(gen_data), valid)
    #         loss_G.backward()
    #         optimizer_G.step()
    #     G.eval()
    #     anomaly_scores = []
    #     for i in range(X.shape[0]):
    #         x = torch.tensor(X[i], dtype=torch.float32, device=device)
    #         z = torch.randn((1, latent_dim), requires_grad=True, device=device)
    #         optimizer_z = optim.Adam([z], lr=0.05)
    #         for _ in range(300):
    #             optimizer_z.zero_grad()
    #             x_gen = G(z)
    #             loss = torch.mean(torch.abs(x_gen - x))
    #             loss.backward()
    #             optimizer_z.step()
    #         with torch.no_grad():
    #             x_gen = G(z)
    #             score = torch.mean(torch.abs(x_gen - x)).item()
    #             anomaly_scores.append(score)
    #     anomaly_scores = np.array(anomaly_scores)
    #     anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-8)
    #     anomaly_scores = np.clip(anomaly_scores, 0, 1)  # Ensure values are in [0, 1]   
    #     return anomaly_scores

    # Main function to detect outliers for each embedding in the dictionary
    def detect_outliers(self, embedding_dict):
        results = {}
        for network_id, channels_dict in embedding_dict.items():
            #start_time = time.time()  # Início da medição do tempo
            results[network_id] = {}
            for channels, embedding_matrix in channels_dict.items():
                node_ids = embedding_matrix[:, 0]
                X = embedding_matrix[:, 1:].astype(float)
                scaler = StandardScaler()
                X_norm = scaler.fit_transform(X)
                scores_isf = self.apply_isolation_forest(X_norm)
                scores_cov = self.apply_elliptic_envelope(X_norm)
                scores_svm = self.apply_sgd_one_class_svm(X_norm)
                scores_os1, scores_os2 = self.calc_os1_os2(X_norm)
                scores_coada = self.calc_coada(X_norm)
                scores_dsvdd = self.calc_svdd(X_norm)
                #scores_anogan = self.calc_anogan(X_norm)
                node_results = {}
                for i, node_id in enumerate(node_ids):
                    node_results[node_id] = {
                        'score_IsF': float(-scores_isf[i]),
                        'score_COV': float(-scores_cov[i]),
                        'score_SVM': float(-scores_svm[i]),
                        'score_OS1': float(scores_os1[i]),
                        'score_OS2': float(scores_os2[i]),
                        'score_CoADA': float(scores_coada[i]),
                        'score_DSVDD': float(scores_dsvdd[i]),
                        #'score_AnoGAN': float(scores_anogan[i])
                    }
                results[network_id][channels] = node_results
            #elapsed = time.time() - start_time
            #print(f"Time spent on the network {network_id}: {elapsed:.2f} seconds")
        return results
    
# Create Outliers DataFrames

class OutlierResultsDFBuilder:
    """
    Converts outlier results (dict) into dictionaries of DataFrames by network/channel.
    """
    def __init__(self, outlier_results):
        self.outlier_results = outlier_results
        self.dfs = {}

    def build(self):
        """
        Builds the dictionary of DataFrames from the outlier results.
        """
        for network_id, channels_dict in self.outlier_results.items():
            self.dfs[network_id] = {}
            for channels, node_results in channels_dict.items():
                # Create a DataFrame for each network/channel with node_id and outlier scores
                df = pd.DataFrame([
                    {
                        "node_id": node_id,
                        "score_IsF": res["score_IsF"],   
                        "score_COV": res["score_COV"],
                        "score_SVM": res["score_SVM"],
                        "score_OS1": res["score_OS1"],
                        "score_OS2": res["score_OS2"],
                        "score_CoADA": res["score_CoADA"],
                        "score_DSVDD": res["score_DSVDD"],
                        #"score_AnoGAN": res["score_AnoGAN"]
                    }
                    for node_id, res in node_results.items()
                ])
                self.dfs[network_id][channels] = df
        return self.dfs

# Create normalized DataFrames 

class OutlierScoreNormalizer:
    """
    Class to normalize and rank outlier scores in DataFrames from multiple methods.
    """
    def __init__(self, dfs_dict, score_cols=None):
        """
        Args:
            dfs_dict (dict): Dictionary {network_id: {channels: DataFrame}}
            score_cols (list or None): List of score columns to process. Default: ['IsF', 'COV', 'SVM']
        """
        self.dfs_dict = dfs_dict
        # Only normalize these columns; others are assumed already normalized
        self.score_cols = score_cols or ['score_IsF', 'score_COV', 'score_SVM']
        self.result = {}

    def assign_descending_and_normalize(self):
        """
        For each DataFrame, replace only 'score_IsF', 'score_COV', 'score_SVM' columns with descending integer ranks,
        then normalize each of these columns to [0, 1]. Other columns remain unchanged.
        Returns a new dictionary.
        """
        result = {}
        for network_id, channels_dict in self.dfs_dict.items():
            result[network_id] = {}
            for channels, df in channels_dict.items():
                df_mod = df.copy()
                n = len(df_mod)
                for col in self.score_cols:
                    if col in df_mod.columns:
                        # Assign descending rank (highest score gets rank 1)
                        order = df_mod[col].rank(method='first', ascending=False).astype(int)
                        df_mod[col] = n - order + 1
                        # Normalize to [0, 1]
                        df_mod[col] = (df_mod[col] - 1) / (n - 1) if n > 1 else 0
                result[network_id][channels] = df_mod
        self.result = result
        return result

# Data Treatment

# Read the edge list from a CSV file using ';' as the separator
edges_df = pd.read_csv(f"{DATA_DIR}/edges.csv", sep=';')

# Create a NetworkX graph from the edge list DataFrame
G = nx.from_pandas_edgelist(edges_df, source='Source', target='Target')

# Identify all connected components in the graph
connected_components = nx.connected_components(G)

# Initialize a DataFrame to store information about each component
component_info = pd.DataFrame(columns=['network', 'nodes', 'edges'])

# Iterate over each connected component in the graph
for idx, component in enumerate(connected_components):
    # Extract the subgraph corresponding to the current component
    subgraph = G.subgraph(component)
    
    # Get the number of nodes and edges in the subgraph
    num_nodes = len(subgraph.nodes)
    num_edges = len(subgraph.edges)
    
    # Append the component information to the DataFrame
    component_info = pd.concat(
        [component_info, pd.DataFrame({'network': [idx+1], 'nodes': [num_nodes], 'edges': [num_edges]})],
        ignore_index=True
    )

# Save the component_info DataFrame to a CSV file without the index
component_info.to_csv(f"{OUTPUT_DIR}/dataset_components.csv", index=False)

# Recommended Network

# Filter components with more than 30 nodes and average degree >= 2
large_components = component_info[
    (component_info['nodes'] >= 30) &
    #(component_info['nodes'] < 40) &
    (2 * component_info['edges'] / (component_info['nodes']) >= 2)
]

large_components.to_csv(f"{OUTPUT_DIR}/dataset_components_large.csv", index=False)

# Save each large network component as an edgelist file

for idx, row in large_components.iterrows():
   network_id = int(row['network'])
   # Get the nodes of the corresponding component (components are indexed from 1)
   component_nodes = list(nx.connected_components(G))[network_id - 1]
   # Create the subgraph corresponding to the component
   subgraph = G.subgraph(component_nodes)
   # Save the subgraph in edgelist format for later use in GNN models
   nx.write_edgelist(subgraph, f"{OUTPUT_DIR}/network_{network_id}.edgelist")

# Build the list of graph components for modeling using the filtered DataFrame and the full graph
graph_components = build_graph_components(large_components, G)

# Embedding Generation

# List of output embedding dimensions to simulate different model complexities
channels_list = [32, 64, 128, 256]
epochs = 100         # Maximum number of training epochs
patience = 5         # Early stopping patience

# Using Graph Attention Networks (GAT)

# Dictionary to store embeddings for each network and channel size
gat_embedding_results = {}  # {network_id: {out_channels: embedding_matrix}}

# Iterate over each large graph component for embedding generation
for gc in graph_components:
    # Get the list of node IDs and map them to indices
    node_list = list(gc.nx_graph.nodes())
    node_idx_map = {node: idx for idx, node in enumerate(node_list)}
    # Convert edges to index-based tuples
    edges = [(node_idx_map[u], node_idx_map[v]) for u, v in gc.nx_graph.edges()]
    if len(edges) == 0:
        continue  # Skip components with no edges

    # Use identity matrix as node features (one-hot encoding)
    x = torch.eye(len(node_list))
    # Create a PyTorch Geometric Data object
    data = Data(x=x, edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous())

    # Initialize dictionary for this network/component
    gat_embedding_results[gc.network_id] = {}

    # Train and evaluate GAT for each embedding dimension
    for out_channels in channels_list:
        # Instantiate the GAT model and optimizer
        model = GAT(in_channels=x.size(1), out_channels=out_channels, num_nodes=x.size(0))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        best_loss = float('inf')  # Track the best loss for early stopping
        best_state = None         # Store the best model state
        patience_counter = 0      # Counter for early stopping

        # Training loop with early stopping
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            # Reconstruct input features from embeddings
            recon = model.reconstruct(data.x, data.edge_index)
            # Compute mean squared error loss
            loss = F.mse_loss(recon, data.x)
            loss.backward()
            optimizer.step()

            # Early stopping: save best model and check patience
            if loss.item() < best_loss - 1e-6:
                best_loss = loss.item()
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break  # Stop training if no improvement

        print(f"Early stopping at epoch {epoch+1} for network {gc.network_id} with {out_channels} channels (GAT)")

        # Load the best model state and generate embeddings
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index).cpu().numpy()

        # Concatenate node IDs with their embeddings for saving
        node_ids = np.array(node_list).reshape(-1, 1)
        embedding_matrix = np.concatenate([node_ids, embeddings], axis=1)
        gat_embedding_results[gc.network_id][out_channels] = embedding_matrix

# Using Graph Transformer (GT)

# Dictionary to store embeddings for each network and channel size
gt_embedding_results = {}  # {network_id: {out_channels: embedding_matrix}}

# Iterate over each large graph component for embedding generation
for gc in graph_components:
    #start_time = time.time()  # Início da medição do tempo
    # Get the list of node IDs and map them to indices
    node_list = list(gc.nx_graph.nodes())
    node_idx_map = {node: idx for idx, node in enumerate(node_list)}
    # Convert edges to index-based tuples
    edges = [(node_idx_map[u], node_idx_map[v]) for u, v in gc.nx_graph.edges()]
    if len(edges) == 0:
        continue  # Skip components with no edges

    # Use identity matrix as node features (one-hot encoding)
    x = torch.eye(len(node_list))
    # Create a PyTorch Geometric Data object
    data = Data(x=x, edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous())

    # Initialize dictionary for this network/component
    gt_embedding_results[gc.network_id] = {}

    # Train and evaluate GraphTransformer for each embedding dimension
    for out_channels in channels_list:
        # Instantiate the GraphTransformer model and optimizer
        model = GraphTransformer(in_channels=x.size(1), out_channels=out_channels, num_nodes=x.size(0))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        best_loss = float('inf')  # Track the best loss for early stopping
        best_state = None         # Store the best model state
        patience_counter = 0      # Counter for early stopping

        # Training loop with early stopping
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            # Reconstruct input features from embeddings
            recon = model.reconstruct(data.x, data.edge_index)
            # Compute mean squared error loss
            loss = F.mse_loss(recon, data.x)
            loss.backward()
            optimizer.step()

            # Early stopping: save best model and check patience
            if loss.item() < best_loss - 1e-6:
                best_loss = loss.item()
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break  # Stop training if no improvement

        print(f"Early stopping at epoch {epoch+1} for network {gc.network_id} with {out_channels} channels (GT)")

        # Load the best model state and generate embeddings
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index).cpu().numpy()

        # Concatenate node IDs with their embeddings for saving
        node_ids = np.array(node_list).reshape(-1, 1)
        embedding_matrix = np.concatenate([node_ids, embeddings], axis=1)
        gt_embedding_results[gc.network_id][out_channels] = embedding_matrix

# Use Outlier Detector:

detector = OutlierDetector()
print("Detecting outliers using IsF, COV, SVM, OS1, OS2, CoADA and DSVDD...")
print("This may take a while, depending on your computer...")

# Detect outliers for GAT embeddings
print("Detecting outliers for GAT embeddings...")
outlier_results_gat = detector.detect_outliers(gat_embedding_results)

print()

# Detect outliers for GT embeddings
print("Detecting outliers for GT embeddings...")
outlier_results_gt = detector.detect_outliers(gt_embedding_results)

# Use for GAT
gat_builder = OutlierResultsDFBuilder(outlier_results_gat)
gat_outlier_dfs = gat_builder.build()

# Use for GT
gt_builder = OutlierResultsDFBuilder(outlier_results_gt)
gt_outlier_dfs = gt_builder.build()

# Use for GAT and GT:
gat_norm = OutlierScoreNormalizer(gat_outlier_dfs)
gat_outlier_dfs_desc_norm = gat_norm.assign_descending_and_normalize()

gt_norm = OutlierScoreNormalizer(gt_outlier_dfs)
gt_outlier_dfs_desc_norm = gt_norm.assign_descending_and_normalize()

# Save Dict Dataframes

# Save the normalized GAT outlier DataFrames dictionary to a pickle file
with open(f"{OUTPUT_DIR}/gat_outlier.pkl", 'wb') as f:
    pickle.dump(gat_outlier_dfs_desc_norm, f)

# Save the normalized GT outlier DataFrames dictionary to a pickle file
with open(f"{OUTPUT_DIR}/gt_outlier.pkl", 'wb') as f:
    pickle.dump(gt_outlier_dfs_desc_norm, f)
## End of script
# Key Roles

# Imports

import pandas as pd
import numpy as np
import pickle
import networkx as nx
import glob
import warnings
warnings.filterwarnings('ignore')
from config import DATA_DIR, OUTPUT_DIR

class KeyRoleLabeler:
    def __init__(self, individuals_df, key_roles_dict, best_models_by_network, gat_scores_dfs, gt_scores_dfs):
        """
        individuals_df: DataFrame with individuals in large components
        key_roles_dict: {network_id: [key_role_ids]}
        best_models_by_network: DataFrame with best model info per network
        gat_scores_dfs, gt_scores_dfs: dicts with scores per network/channel
        """
        self.individuals_df = individuals_df.copy()
        self.key_roles_dict = key_roles_dict
        self.best_models_by_network = best_models_by_network
        self.gat_scores_dfs = gat_scores_dfs
        self.gt_scores_dfs = gt_scores_dfs

    def label_key_roles(self):
        # Initialize 'key' column with 0
        self.individuals_df['key'] = 0
        # Mark key roles for each network
        for network_id, key_roles in self.key_roles_dict.items():
            key_roles_set = set(key_roles)
            mask = self.individuals_df['id'].isin(key_roles_set)
            self.individuals_df.loc[mask, 'key'] = 1

    def add_scores(self):
        # Initialize 'scores' column with NaN
        self.individuals_df['scores'] = np.nan
        # Add scores from the best model for each network
        for _, row in self.best_models_by_network.iterrows():
            network_id = row['network']
            graph = row['graph']
            channels = row['channels']
            model = row['model']
            scores_dict = self.gat_scores_dfs if graph == 'GAT' else self.gt_scores_dfs
            scores_df = scores_dict.get(network_id, {}).get(channels, None)
            if scores_df is None:
                continue
            score_key = f'score_{model}'
            if score_key not in scores_df.columns:
                continue
            aux = scores_df[['node_id', score_key]].rename(columns={'node_id': 'id', score_key: 'scores'})
            mask = self.individuals_df['id'].isin(aux['id'])
            self.individuals_df.loc[mask, 'scores'] = self.individuals_df.loc[mask].merge(
                aux, on='id', how='left')['scores_y'].values

    def process(self):
        self.label_key_roles()
        self.add_scores()
        return self.individuals_df

# Read Data

# Load summary results for GAT from CSV
gat_summary_disrupt = pd.read_csv(f"{OUTPUT_DIR}/gat_summary_disrupt.csv")

# Load summary results for GT from CSV
gt_summary_disrupt = pd.read_csv(f"{OUTPUT_DIR}/gt_summary_disrupt.csv")

# Load the dictionaries of DataFrames directly from pickle files
with open(f"{OUTPUT_DIR}/gat_outlier.pkl", 'rb') as f:
    gat_scores_dfs = pickle.load(f)  # Dictionary: {network_id: {channels: DataFrame}}

with open(f"{OUTPUT_DIR}/gt_outlier.pkl", 'rb') as f:
    gt_scores_dfs = pickle.load(f)  # Dictionary: {network_id: {channels: DataFrame}}

individuals_df = pd.read_csv(f"{DATA_DIR}/individuals.csv", sep=';')

# Results 
# Selects the best model per network and channels according to the defined criteria:
# 1. Highest 'number_of_components'
# 2. Lowest 'global_efficiency'
# 3. Lowest 'largest_component_nodes'
gat_best_models = (
    gat_summary_disrupt
    .sort_values(['network', 'channels', 'number_of_components', 'global_efficiency', 'largest_component_nodes'],
                 ascending=[True, True, False, True, True])
    .groupby(['network', 'channels'], as_index=False)
    .first()
)

gat_best_models_by_network = (
    gat_best_models
    .sort_values(['network', 'steps_to_disruption', 'number_of_components', 'global_efficiency', 'largest_component_nodes'],
                 ascending=[True, True, False, True, True])
    .groupby('network', as_index=False)
    .first()
)

gt_best_models = (
    gt_summary_disrupt
    .sort_values(['network', 'channels', 'number_of_components', 'global_efficiency', 'largest_component_nodes'],
                 ascending=[True, True, False, True, True])
    .groupby(['network', 'channels'], as_index=False)
    .first()
)

gt_best_models_by_network = (
    gt_best_models
    .sort_values(['network', 'steps_to_disruption', 'number_of_components', 'global_efficiency', 'largest_component_nodes'],
                 ascending=[True, True, False, True, True])
    .groupby('network', as_index=False)
    .first()
)

# Copy the best models by network for GAT and add a column indicating the graph type
gat_best = gat_best_models_by_network.copy()
gat_best['graph'] = 'GAT'

# Copy the best models by network for GT and add a column indicating the graph type
gt_best = gt_best_models_by_network.copy()
gt_best['graph'] = 'GT'

# Concatenate GAT and GT best models into a single DataFrame
combined = pd.concat([gat_best, gt_best], ignore_index=True)

# Add the 'num_nodes' column with the number of nodes for each network
nodes_count = []
for _, row in combined.iterrows():
    graph = row['graph']
    network = row['network']
    channels = row['channels']
    # Select the correct scores dictionary based on graph type
    scores_dict = gat_scores_dfs if graph == 'GAT' else gt_scores_dfs
    scores_df = scores_dict.get(network, {}).get(channels, None)
    if scores_df is not None:
        nodes_count.append(len(scores_df))
    else:
        nodes_count.append(np.nan)
combined['num_nodes'] = nodes_count

# Sort by number of nodes (descending)
combined_sorted = combined.sort_values(
    ['num_nodes', 'network', 'steps_to_disruption', 'number_of_components', 'global_efficiency', 'largest_component_nodes'],
    ascending=[False, True, True, False, True, True]
)

# Select the best model per network based on the criteria, but keep the order by num_nodes
best_models_by_network = (
    combined_sorted
    .groupby('network', as_index=False)
    .first()
    .sort_values('num_nodes', ascending=False)  # Final sorting by num_nodes
    .loc[:, ['graph', 'num_nodes'] + [col for col in gat_best_models_by_network.columns]]
)

# Save the resulting DataFrame to CSV
best_models_by_network.to_csv(f"{OUTPUT_DIR}/best_models_by_network.csv", sep=';', index=False)

# Labeling Key Roles

# Path to the saved edgelist files
edgelist_files = glob.glob(f"{OUTPUT_DIR}/network_*.edgelist")

# Dictionary to store the loaded subgraphs
subgraphs = {}

for filepath in edgelist_files:
    # Extract the network_id from the filename
    network_id = int(filepath.split("_")[-1].split(".")[0])
    # Read the subgraph from the edgelist file
    G_sub = nx.read_edgelist(filepath, nodetype=str)  # or nodetype=int, depending on your node type
    subgraphs[network_id] = G_sub

# Get all nodes present in the edgelist files (i.e., in the main subgraphs)
nodes_in_large_components = set()
for G in subgraphs.values():
    nodes_in_large_components.update(G.nodes())

# Filter the dataframe for nodes that are in the large components
individuals_large_components = individuals_df[individuals_df['id'].isin(nodes_in_large_components)].copy()
individuals_other_components = individuals_df[~individuals_df['id'].isin(nodes_in_large_components)].copy()

# Identify 'key roles' (most important nodes) for each network, excluding the ids removed in steps_to_disruption

key_roles_dict = {}

for _, row in best_models_by_network.iterrows():
    graph = row['graph']
    network = row['network']
    channels = row['channels']
    model = row['model']
    steps_to_disruption = row['steps_to_disruption']

    # Select the correct scores dictionary (GAT or GT)
    scores_dict = gat_scores_dfs if graph == 'GAT' else gt_scores_dfs
    scores_df = scores_dict.get(network, {}).get(channels, None)
    if scores_df is None:
        continue

    score_key = f'score_{model}'
    if score_key not in scores_df.columns:
        continue

    # Sort nodes by the model's score (descending)
    sorted_nodes = scores_df[['node_id', score_key]].sort_values(score_key, ascending=False)
    # Get the ids of the nodes removed during disruption (top N)
    removed_ids = set(sorted_nodes.head(steps_to_disruption)['node_id'])
    # The key roles are the next best nodes not removed in disruption
    remaining_nodes = sorted_nodes[~sorted_nodes['node_id'].isin(removed_ids)]
    key_roles = remaining_nodes.head(steps_to_disruption)['node_id'].tolist()

    key_roles_dict[network] = key_roles

# Usage:
labeler = KeyRoleLabeler(
    individuals_large_components,
    key_roles_dict,
    best_models_by_network,
    gat_scores_dfs,
    gt_scores_dfs
)
individuals_large_components_with_key = labeler.process()

# Count how many key roles exist in individuals_large_components_with_key
num_key_roles = individuals_large_components_with_key['key'].sum()
print(f"key roles: {num_key_roles}")

# Save the dataframe 'individuals_large_components_with_key' to a CSV file
individuals_large_components_with_key.to_csv(f"{OUTPUT_DIR}/labeled_individuals_dataset.csv", sep=';', index=False, header=True)

# Save the dataframe 'individuals_other_components' to a CSV file
individuals_other_components.to_csv(f"{OUTPUT_DIR}/individuals_other_components_dataset.csv", sep=';', index=False, header=True)
## End of script
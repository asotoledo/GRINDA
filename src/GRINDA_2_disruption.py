# Disruption

# Imports

import pandas as pd
import numpy as np
import networkx as nx
import pickle
import warnings
warnings.filterwarnings('ignore')
from config import DATA_DIR, OUTPUT_DIR

def set_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

class OutlierRankingTable:
    """
    Builds ranking tables for each network and channel, storing ordered node_ids by outlier score.
    """
    def __init__(self, dfs_dict, score_cols=None):
        """
        Args:
            dfs_dict (dict): {network_id: {channels: DataFrame}}
                Dictionary where each key is a network_id, and each value is another dictionary
                mapping channel sizes to DataFrames containing outlier scores for each node.
            score_cols (list): List of score columns to use for ranking. If None, uses all supported methods.
        """
        self.dfs_dict = dfs_dict
        # List of score columns to use for ranking. If not provided, use all available methods.
        self.score_cols = score_cols or [
            'score_IsF', 'score_COV', 'score_SVM', 'score_OS1', 'score_OS2',
            'score_CoADA', 'score_DSVDD'
        ]
        # This will store the ranking tables: {network_id: {channels: DataFrame}}
        self.tables = {}

    def build(self):
        """
        For each network/channel, creates a DataFrame with columns as score names and values as ordered node_ids.
        Each column contains the node_ids sorted by descending outlier score for that method.
        """
        for network_id, channels_dict in self.dfs_dict.items():
            self.tables[network_id] = {}
            for channels, df in channels_dict.items():
                ranking_dict = {}
                for col in self.score_cols:
                    if col in df.columns:
                        # Sort the DataFrame by the current score column in descending order
                        ordered_nodes = df.sort_values(col, ascending=False)['node_id'].tolist()
                        # Remove possible duplicates and keep order (should not be necessary, but for safety)
                        ranking_dict[col.replace('score_', '')] = ordered_nodes
                # Find the maximum list length among all ranking lists (should be equal to number of nodes)
                max_len = max(len(lst) for lst in ranking_dict.values())
                for k in ranking_dict:
                    # Pad lists with None so all columns have the same length (for DataFrame construction)
                    ranking_dict[k] += [None] * (max_len - len(ranking_dict[k]))
                # Create a DataFrame where each column is a ranking for a method, rows are ranks (top to bottom)
                self.tables[network_id][channels] = pd.DataFrame(ranking_dict)
        return self.tables

    def get_table(self, network_id, channels):
        """
        Returns the ranking DataFrame for a specific network and channel.
        Args:
            network_id (int): The network/component identifier.
            channels (int): The embedding dimension/channel size.
        Returns:
            pd.DataFrame or None: The ranking table for the given network/channel, or None if not found.
        """
        return self.tables.get(network_id, {}).get(channels, None)
    
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

class NetworkDisruptionSimulator:
    """
    Simulates network disruption for different methods and channels, saving the results of each step.
    """
    def __init__(self, graph_components, ranking_tables, data_network):
        self.graph_components = graph_components
        self.ranking_tables = ranking_tables
        self.data_network = data_network
        self.models = ['IsF', 'COV', 'SVM', 'OS1', 'OS2', 'CoADA', 'DSVDD']
        self.results_steps = {}
        self.summary = []

    @staticmethod
    def create_graph(df_edges):
        return nx.from_pandas_edgelist(df_edges, source='Source', target='Target')

    @staticmethod
    def components_graph(graph):
        num_components = nx.number_connected_components(graph)
        largest_component_size = len(max(nx.connected_components(graph), key=len))
        return num_components, largest_component_size

    @staticmethod
    def remodel_node(G, node_key):
        new_G = G.copy()
        new_G.remove_node(node_key)
        return new_G

    @staticmethod
    def normalize_vector(vector):
        vector = np.array(vector)
        return (vector - np.min(vector)) / (np.ptp(vector) if np.ptp(vector) != 0 else 1)

    @staticmethod
    def find_position(vector):
        return int(np.where(vector == np.max(vector))[0][-1])

    def simulate(self, channels_list):
        print("Networks to be processed:", [gc.network_id for gc in self.graph_components])
        for gc in self.graph_components:
            network_id = gc.network_id
            for channels in channels_list:
                if network_id not in self.ranking_tables or channels not in self.ranking_tables[network_id]:
                    continue
                ranking_df = self.ranking_tables[network_id][channels]
                for model in self.models:
                    if model not in ranking_df.columns:
                        continue
                    key = ranking_df[model].tolist()
                    G = self.create_graph(self.data_network[network_id])
                    steps = [0]
                    components = []
                    size_largest_component = []
                    global_efficiency = []
                    nodes_count = []
                    edges_count = []
                    num_components, largest_component = self.components_graph(G)
                    components.append(num_components)
                    size_largest_component.append(largest_component)
                    global_efficiency.append(nx.global_efficiency(G))
                    nodes_count.append(G.number_of_nodes())
                    edges_count.append(G.number_of_edges())
                    H = G.copy()
                    for n, node in enumerate(key[:-1]):
                        H = self.remodel_node(H, node)
                        num_components, largest_component = self.components_graph(H)
                        components.append(num_components)
                        size_largest_component.append(largest_component)
                        global_efficiency.append(nx.global_efficiency(H))
                        nodes_count.append(H.number_of_nodes())
                        edges_count.append(H.number_of_edges())
                        steps.append(n + 1)
                    normalized_components = self.normalize_vector(components)
                    normalized_size_largest_component = self.normalize_vector(size_largest_component)
                    normalized_global_efficiency = self.normalize_vector(global_efficiency)
                    self.results_steps[(network_id, channels, model)] = {
                        'steps': steps,
                        'components': components,
                        'global_efficiency': global_efficiency,
                        'size_largest_component': size_largest_component,
                        'normalized_components': normalized_components,
                        'normalized_global_efficiency': normalized_global_efficiency,
                        'normalized_size_largest_component': normalized_size_largest_component,
                        'nodes_count': nodes_count,
                        'edges_count': edges_count
                    }
                    position = self.find_position(components)
                    summary_row = {
                        'network': network_id,
                        'channels': channels,
                        'model': model,
                        'steps_to_disruption': position,
                        'number_of_components': components[position],
                        'global_efficiency': global_efficiency[position],
                        'largest_component_nodes': size_largest_component[position],
                        'nodes_count': nodes_count[position],
                        'edges_count': edges_count[position]
                    }
                    self.summary.append(summary_row)
                    print(f"Network: {network_id} | Channels: {channels} | Model: {model}")
                    print(f"  Steps to disruption: {position}")
                    print(f"  Number of components: {components[position]}")
                    print(f"  Global Efficiency: {global_efficiency[position]:.6f}")
                    print(f"  Number of nodes in largest component: {size_largest_component[position]}")
                    print("-" * 60)

    def get_steps_results(self):
        # Return all detailed step results
        return self.results_steps

    def get_summary(self):
        # Return the summary as a DataFrame
        return pd.DataFrame(self.summary)

# Read data 

# Load detailed disruption simulation results (steps) for GAT using pickle
with open(f"{OUTPUT_DIR}/gat_outlier.pkl", 'rb') as f:
    gat_outlier = pickle.load(f)

# Load detailed disruption simulation results (steps) for GT using pickle
with open(f"{OUTPUT_DIR}/gt_outlier.pkl", 'rb') as f:
    gt_outlier = pickle.load(f)

    # Read the edge list from a CSV file using ';' as the separator

edges_df = pd.read_csv(f"{DATA_DIR}/edges.csv", sep=';')

# Create a NetworkX graph from the edge list DataFrame
G = nx.from_pandas_edgelist(edges_df, source='Source', target='Target')

large_components = pd.read_csv(f"{OUTPUT_DIR}/dataset_components_large.csv")

# Nodes Ranking

# Use for GAT and GT:
# Build ranking tables for GAT outlier results (normalized and ranked)
gat_ranking = OutlierRankingTable(gat_outlier)
gat_ranking_tables = gat_ranking.build()

# Build ranking tables for GT outlier results (normalized and ranked)
gt_ranking = OutlierRankingTable(gt_outlier)
gt_ranking_tables = gt_ranking.build()

# Save detailed nodes ranking simulation for GAT using pickle
# Pickle is a Python library for serializing and deserializing Python objects to and from byte streams
with open(f"{OUTPUT_DIR}/gat_ranking.pkl", 'wb') as f:
    pickle.dump(gat_ranking_tables, f)

# Save detailed nodes ranking simulation for GT using pickle
# Pickle is a Python library for serializing and deserializing Python objects to and from byte streams
with open(f"{OUTPUT_DIR}/gt_ranking.pkl", 'wb') as f:
    pickle.dump(gt_ranking_tables, f)

# Build the list of graph components for modeling using the filtered DataFrame and the full graph
graph_components = build_graph_components(large_components, G)

# Create a dictionary to store the edge list DataFrames for each network
data_network = {}

for gc in graph_components:
    # Extract the list of edges from the component's graph
    edges = list(gc.nx_graph.edges())
    # Create the edge list DataFrame
    df_edges = pd.DataFrame(edges, columns=['Source', 'Target'])
    # Save it in the dictionary using the network_id as the key
    data_network[gc.network_id] = df_edges

# Network disruption simulation

# Use for GAT:
print("Simulating network disruption for GAT...")
simulator_gat = NetworkDisruptionSimulator(graph_components, gat_ranking_tables, data_network)
channels_list = [32, 64, 128, 256]
simulator_gat.simulate(channels_list)
gat_steps_results = simulator_gat.get_steps_results()
gat_summary = simulator_gat.get_summary()

# Save detailed disruption simulation results (steps) for GAT using pickle
# Pickle is a Python library for serializing and deserializing Python objects to and from byte streams
with open(f"{OUTPUT_DIR}/gat_steps_disrupt.pkl", 'wb') as f:
    pickle.dump(gat_steps_results, f)

# Save the summary results for GAT as a CSV file
gat_summary.to_csv(f"{OUTPUT_DIR}/gat_summary_disrupt.csv", index=False)

# Use for GT:
print("Simulating network disruption for ...")
simulator_gt = NetworkDisruptionSimulator(graph_components, gt_ranking_tables, data_network)
channels_list = [32, 64, 128, 256]
simulator_gt.simulate(channels_list)
gt_steps_results = simulator_gt.get_steps_results()
gt_summary = simulator_gt.get_summary()

# Save detailed disruption simulation results (steps) for GT using pickle
with open(f"{OUTPUT_DIR}/gt_steps_disrupt.pkl", 'wb') as f:
    pickle.dump(gt_steps_results, f)

# Save the summary results for GT as a CSV file
gt_summary.to_csv(f"{OUTPUT_DIR}/gt_summary_disrupt.csv", index=False)
# Save the summary results for GT as a CSV file
gt_summary.to_csv(f"{OUTPUT_DIR}/gt_summary_disrupt.csv", index=False)
## End of script
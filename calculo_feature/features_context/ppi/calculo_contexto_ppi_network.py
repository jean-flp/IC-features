# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 13:17:18 2025

@author: Emanu
"""
#%%
import pandas as pd
import networkx as nx
import os
import time
from collections import OrderedDict
import math

#%%
seconds_ini = time.time()
calculo_feature_dir = os.getcwd()

path_data_processed = os.path.abspath(
    os.path.join(
        os.path.join(calculo_feature_dir, "../../../"),
        "data/processed/context_features/ppi_network"
    )
)

path_data_raw = os.path.abspath(
    os.path.join(
        os.path.join(calculo_feature_dir, "../../../"),
        "data/raw/string/ppi"
    )
)
print(path_data_processed)
print(path_data_raw)

#%%
def mapProtein(protein_interaction_df, protein_map):
    protein_interation_masked = pd.DataFrame()
    proteins = ['protein1', 'protein2']
    
    for p in proteins:
        protein_interation_masked[p] = protein_interaction_df[p].map(protein_map)
    
    return protein_interation_masked

#%%
def generateGraph(protein_interation_masked):
    return nx.from_pandas_edgelist(
        protein_interation_masked,
        source="protein1",
        target="protein2"
    )

#%%
def generateDF(protein_interaction_graph):
    return nx.to_pandas_edgelist(protein_interaction_graph)

#%%
def local_average_connectivity(graph):
    lac_values = {}
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))

        if not neighbors:
            lac_values[node] = 0.0
            continue

        neighbor_degrees = [graph.degree(n) for n in neighbors]
        lac_values[node] = sum(neighbor_degrees) / len(neighbor_degrees)

    values = list(lac_values.values())
    min_lac = min(values)
    max_lac = max(values)

    for node in lac_values:
        if max_lac - min_lac == 0:
            lac_values[node] = 0.0
        else:
            lac_values[node] = (lac_values[node] - min_lac) / (max_lac - min_lac)

    return lac_values

#%%
def edge_clustering_coefficient(graph):
    edge_cc = {}
    for u, v in graph.edges():
        common_neighbors = set(graph.neighbors(u)).intersection(set(graph.neighbors(v)))
        num_triangles = len(common_neighbors)

        k_u = graph.degree(u)
        k_v = graph.degree(v)

        if min(k_u - 1, k_v - 1) == 0:
            C_uv = 0
        else:
            C_uv = num_triangles / min(k_u - 1, k_v - 1)

        edge_cc[tuple(sorted((u, v)))] = C_uv

    nc = {}
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            nc[node] = 0.0
            continue

        total = 0
        for neighbor in neighbors:
            edge = tuple(sorted((node, neighbor)))
            total += edge_cc.get(edge, 0.0)

        nc[node] = total / len(neighbors)

    return nc

#%%
def dmnc(graph):
    densities = {}

    for node in graph.nodes():
        neighbors = set(graph.neighbors(node))
        if not neighbors:
            densities[node] = 0.0
            continue

        neighborhood_subgraph = graph.subgraph(neighbors)
        components = list(nx.connected_components(neighborhood_subgraph))
        largest_component = max(components, key=len)

        mnc = graph.subgraph(largest_component)
        n = mnc.number_of_nodes()
        e = mnc.number_of_edges()

        if n <= 1:
            density = 0.0
        else:
            density = (2 * e) / (n * (n - 1))

        densities[node] = density

    return densities

#%%
def topology_potential(graph, sigma=0.9428):
    tp_values = {}
    shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))

    for i in graph.nodes():
        tp = 0.0
        for j in graph.nodes():
            if i == j:
                continue
            d = shortest_paths[i].get(j, float('inf'))
            if d < float('inf'):
                tp += math.exp(- (d / sigma) ** 2)
        tp_values[i] = tp

    return tp_values

#%%
for organismo in os.listdir(path_data_raw):
    pasta_organismo = os.path.join(path_data_raw, organismo)
    nome_arquivo = os.listdir(pasta_organismo)[0]
    path_arquivo = os.path.join(pasta_organismo, nome_arquivo)

    data = pd.read_csv(path_arquivo, sep=" ")

    protein_map = {
        v: k for k, v in enumerate(
            set(data["protein1"]).union(set(data["protein2"]))
        )
    }

    protein_interaction_masked = mapProtein(data, protein_map)
    graph = generateGraph(protein_interaction_masked)

    df_graph = generateDF(graph)

    # Centralidades básicas
    degree = nx.degree_centrality(graph)
    eigenvector = nx.eigenvector_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, k=380)
    clustering = nx.clustering(graph)

    # Closeness
    closeness = {}
    for i in range(len(protein_map)):
        closeness[i] = nx.closeness_centrality(graph, u=i)

    # Features custom
    lac = local_average_connectivity(graph)
    print(f'Terminou LAC do {organismo}')

    nc = edge_clustering_coefficient(graph)
    print(f'Terminou NC do {organismo}')

    dmnc_values = dmnc(graph)  # ✅ corrigido aqui
    print(f'Terminou DMNC do {organismo}')

    tp = topology_potential(graph)
    print(f'Terminou tp do {organismo}')

    # Ordenação
    degree_ordered = OrderedDict(sorted(degree.items()))
    eigenvector_ordered = OrderedDict(sorted(eigenvector.items()))
    betweenness_ordered = OrderedDict(sorted(betweenness.items()))
    closeness_ordered = OrderedDict(sorted(closeness.items()))
    clustering_ordered = OrderedDict(sorted(clustering.items()))
    lac_ordered = OrderedDict(sorted(lac.items()))
    nc_ordered = OrderedDict(sorted(nc.items()))
    dmnc_ordered = OrderedDict(sorted(dmnc_values.items()))  # ✅ corrigido
    tp_ordered = OrderedDict(sorted(tp.items()))

    # DataFrame final
    protein_features = pd.concat([
        pd.Series(list(protein_map.keys())),
        pd.Series(list(degree_ordered.values())),
        pd.Series(list(eigenvector_ordered.values())),
        pd.Series(list(betweenness_ordered.values())),
        pd.Series(list(closeness_ordered.values())),
        pd.Series(list(lac_ordered.values())),
        pd.Series(list(nc_ordered.values())),
        pd.Series(list(dmnc_ordered.values())),
        pd.Series(list(tp_ordered.values())),
        pd.Series(list(clustering_ordered.values()))
    ], axis=1)

    protein_features.columns = [
        "Protein_key",
        "DegreeCentrality",
        "EigenvectorCentrality",
        "BetweennessCentrality",
        "ClosenessCentrality",
        "LocalAverageConnectivity",
        "NC",
        "DMNC",
        "TP",
        "Clustering"
    ]

    nome_arquivo_destino = nome_arquivo.strip('.txt')

    path_result = os.path.abspath(
        os.path.join(
            path_data_processed,
            f'feature.contexto.network.{organismo}.{nome_arquivo_destino}.tsv'
        )
    )

    protein_features.to_csv(path_result, sep=' ', index=False)

#%%
seconds_fini = time.time()
print("Seconds since epoch =", seconds_fini - seconds_ini)
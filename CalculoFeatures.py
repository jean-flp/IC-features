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
cdir = os.getcwd()
dados_STRING = r"STRING_data"
resultados = r"resultados"
caminho_dados = os.path.join(cdir, dados_STRING)
caminho_resultados = os.path.join(cdir, resultados)

#%%
def mapProtein(protein_interaction_df, protein_map):
    
    protein_interation_masked = pd.DataFrame()
    
    proteins = ['protein1', 'protein2']
    
    for p in proteins:
        protein_interation_masked[p] = protein_interaction_df[p].map(protein_map)
    
    return protein_interation_masked

#%%
def generateGraph(protein_interation_masked):

    # Gerar grafo a partir do mapeamento do dataframe do Pandas
    protein_interaction_graph = nx.from_pandas_edgelist(
        protein_interation_masked, 
        source = "protein1", 
        target = "protein2"
    )
    
    return protein_interaction_graph

#%%
def generateDF(protein_interaction_graph):
    return nx.to_pandas_edgelist(protein_interaction_graph)


#%%
""" Cálculo de Local Average Connectivity """
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
        normalized_lac = (lac_values[node] - min_lac) / (max_lac - min_lac)
        lac_values[node] = normalized_lac

    return lac_values

#%%   
""" Cálculo do Edge Clustering Coefficient Centrality """    
def edge_clustering_coefficient(graph):
    # Calculo de ECC para cada vértice
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

    # Somatório de ecc para calcular o NC
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
""" Cálculo de Density of Maximum Neighborhood Component """
def dmnc(graph):
    densities = {}

    for node in graph.nodes():
        neighbors = set(graph.neighbors(node))
        if not neighbors:
            densities[node] = 0.0
            continue

        # Subgraph of neighbors
        neighborhood_subgraph = graph.subgraph(neighbors)

        # Find connected components and select the largest one
        components = list(nx.connected_components(neighborhood_subgraph))
        largest_component = max(components, key=len)

        mnc = graph.subgraph(largest_component)
        n = mnc.number_of_nodes()
        e = mnc.number_of_edges()

        # Calculate density
        if n <= 1:
            density = 0.0
        else:
            density = (2 * e) / (n * (n - 1))

        densities[node] = density

    return densities

#%%
""" Cálculo de Topology Potential-Based Centrality """
def topology_potential(graph, sigma=0.9428):
    tp_values = {}

    # Compute all pairs shortest path lengths
    shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))

    for i in graph.nodes():
        tp = 0.0
        for j in graph.nodes():
            if i == j:
                continue
            d = shortest_paths[i].get(j, float('inf'))  # If disconnected
            if d < float('inf'):
                tp += math.exp(- (d / sigma) ** 2)
        tp_values[i] = tp

    return tp_values
#%%
for arquivo_STRING in os.listdir(caminho_dados):
    
    data = pd.read_csv(os.path.join(caminho_dados, arquivo_STRING), sep = " ")
    protein_map = { v:k for k, v in enumerate(set(data.loc[:, "protein1"]).union(
    set(data.loc[:, "protein2"]))) }
    
    protein_interaction_masked = mapProtein(data, protein_map)
    graph = generateGraph(protein_interaction_masked)
    
    df_graph = generateDF(graph)
    
    """ Cálculo de Grau (Degree), Eigenvector, Betweenness, Subgraph, Clustering """
    degree = nx.degree_centrality(graph)
    eigenvector = nx.eigenvector_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, k=380)
    clustering = nx.clustering(graph)
    
    """ Closeness """
    closeness = {}
    for i in range(len(protein_map)):
        closeness_tmp = nx.closeness_centrality(graph, u=i)
        closeness[i] = closeness_tmp
        
    lac = local_average_connectivity(graph)
    print(f'Terminou LAC do {arquivo_STRING}')
    nc = edge_clustering_coefficient(graph)
    print(f'Terminou NC do {arquivo_STRING}')
    dmnc = dmnc(graph)
    print(f'Terminou DMNC do {arquivo_STRING}')
    tp = topology_potential(graph)
    print(f'Terminou tp do {arquivo_STRING}')
        
    degree_ordered = OrderedDict(sorted(degree.items()))
    eigenvector_ordered = OrderedDict(sorted(eigenvector.items()))
    betweenness_ordered = OrderedDict(sorted(betweenness.items()))
    closeness_ordered = OrderedDict(sorted(closeness.items()))
    clustering_ordered = OrderedDict(sorted(clustering.items()))
    lac_ordered = OrderedDict(sorted(lac.items()))
    nc_ordered = OrderedDict(sorted(nc.items()))
    dmnc_ordered = OrderedDict(sorted(dmnc.items()))
    tp_ordered = OrderedDict(sorted(tp.items()))
    
    
    protein_features = pd.concat([pd.Series(list(protein_map.keys())),
                                  pd.Series(list(degree_ordered.values())), 
                                  pd.Series(list(eigenvector_ordered.values())),
                                  pd.Series(list(betweenness_ordered.values())), 
                                  pd.Series(list(closeness_ordered.values())),
                                  pd.Series(list(lac_ordered.values())),
                                  pd.Series(list(nc_ordered.values())),
                                  pd.Series(list(dmnc_ordered.values())),
                                  pd.Series(list(tp_ordered.values())),
                                  pd.Series(list(clustering_ordered.values()))], axis=1)
    
    protein_features.columns = ["Protein_key",
                                "DegreeCentrality",
                                "EigenvectorCentrality",
                                "BetweennessCentrality",
                                "ClosenessCentrality",
                                "LocalAverageConnectivity",
                                "NC",
                                "DMNC",
                                "TP",
                                "Clustering"]
    
    nome_arquivo = arquivo_STRING.strip('.txt')
    protein_features.to_csv(f'{nome_arquivo}_resultado.tsv', sep = ' ', index=False)

#%%
# Termnina primeira parte do código e começa a segunda com outros tipos de features
seconds_fini = time.time()
print("Seconds since epoch =", seconds_fini - seconds_ini)
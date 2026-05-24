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
                os.path.join(calculo_feature_dir,"../../../"),
                "data/processed/context_features/ppi_network"
            ) 
    )

path_data_raw = os.path.abspath(
    os.path.join(
                os.path.join(calculo_feature_dir,"../../../"),
                "data/raw/string/ppi"
            ) 
)

#%%
def mapProtein(protein_interaction_df, protein_map):
    return protein_interaction_df[['protein1', 'protein2']].apply(
        lambda col: col.map(protein_map)
    )

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
        if neighbors:
            lac_values[node] = sum(dict(graph.degree(neighbors)).values()) / len(neighbors)
        else:
            lac_values[node] = 0.0

    values = list(lac_values.values())
    min_lac, max_lac = min(values), max(values)

    if max_lac == min_lac:
        return {node: 0.0 for node in lac_values}

    return {
        node: (val - min_lac) / (max_lac - min_lac)
        for node, val in lac_values.items()
    }

#%%   
   
def edge_clustering_coefficient(graph):
    neighbors = {n: set(graph.neighbors(n)) for n in graph.nodes()}
    degrees = dict(graph.degree())

    edge_cc = {}

    for u, v in graph.edges():
        common = neighbors[u] & neighbors[v]
        denom = min(degrees[u] - 1, degrees[v] - 1)

        edge_cc[(u, v) if u < v else (v, u)] = (
            len(common) / denom if denom > 0 else 0.0
        )

    nc = {}
    for node in graph.nodes():
        neigh = neighbors[node]
        if not neigh:
            nc[node] = 0.0
            continue

        total = sum(
            edge_cc[(node, n) if node < n else (n, node)]
            for n in neigh
        )
        nc[node] = total / len(neigh)

    return nc

#%%
def dmnc(graph):
    densities = {}

    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            densities[node] = 0.0
            continue

        sub = graph.subgraph(neighbors)
        largest_cc = max(nx.connected_components(sub), key=len)

        mnc = sub.subgraph(largest_cc)
        n = mnc.number_of_nodes()
        e = mnc.number_of_edges()

        densities[node] = (2 * e) / (n * (n - 1)) if n > 1 else 0.0

    return densities

#%%
def topology_potential(graph, sigma=0.9428):
    tp_values = {}

    for i, lengths in nx.all_pairs_shortest_path_length(graph):
        tp = 0.0
        for j, d in lengths.items():
            if i != j:
                tp += math.exp(- (d / sigma) ** 2)
        tp_values[i] = tp

    return tp_values

#%%
for organismo in os.listdir(path_data_raw):
    
    pasta_organismo = os.path.join(path_data_raw,organismo)

    #precisa do 0 pois será o primeiro arquivo da pasta, de preferencia somente haverá um 
    # unico arquivo pois pela estruturação deveria haver um unico.
    nome_arquivo = os.listdir(pasta_organismo)[0]

    path_arquivo = os.path.join(pasta_organismo, nome_arquivo)

    data = pd.read_csv(path_arquivo,sep = " ")
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
    print(f'Terminou LAC do {organismo}')
    nc = edge_clustering_coefficient(graph)
    print(f'Terminou NC do {organismo}')
    dmnc = dmnc(graph)
    print(f'Terminou DMNC do {organismo}')
    tp = topology_potential(graph)
    print(f'Terminou tp do {organismo}')
        
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
    nome_arquivo_destino = nome_arquivo.strip('.txt')
    path_result =  os.path.abspath(
                                        os.path.join(
                                                    os.path.join(path_data_processed),
                                                    f'feature_contexto_network_{organismo}_{nome_arquivo_destino}.tsv'
                                                ) 
                                    )
    protein_features.to_csv(path_result, sep = ' ', index=False)
# Termnina primeira parte do código e começa a segunda com outros tipos de features
seconds_fini = time.time()
print("Seconds since epoch =", seconds_fini - seconds_ini)
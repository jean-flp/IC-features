#%%
import os
import time
import pandas as pd

#%%
seconds_ini = time.time()

calculo_feature_dir = os.getcwd()

path_data_processed = os.path.abspath(
    os.path.join(calculo_feature_dir, "../data/processed/")
)

final_path = os.path.join(
    path_data_processed,
    'synthesized'
)
print(calculo_feature_dir)
print(path_data_processed)
print(final_path)

#%%
dict_df = {}

arquivos_ignorados = {
    "global.embedding.esm.tsv",
    "global.embedding.prot.tsv",
    "global.embedding.proteinbert.tsv"
}

#%%
dict_organismo = {
    '6239': 'cel',
    '6183': 'man',
    '10090': 'mus',
    '4932': 'sce',
    '7227': 'dme'
}

#%%
dict_features_merged = {
    '6239': None,
    '6183': None,
    '10090': None,
    '4932': None,
    '7227': None
}

#%%
possible_keys = [
    'GeneID',
    'Protein_key',
    'Locus',
    'protein_id'
]

#%%
def is_all_embedding_file(file_name: str) -> bool:

    file_name = file_name.lower()

    return (
        file_name.startswith("all.")
        and ".embedding.pca." in file_name
        and (
            ".esm" in file_name
            or ".prot" in file_name
            or ".proteinbert" in file_name
        )
    )

#%%
def split_embedding_by_organism(df):

    """
    Divide embeddings ALL em dataframes separados por organismo.

    Exemplo:
    protein_id = 6183.xxx -> organismo 6183
    protein_id = 10090.xxx -> organismo 10090
    """

    if "protein_id" not in df.columns:
        return {}

    result = {}

    for organism_id in dict_organismo.keys():

        mask = (
            df["protein_id"]
            .astype(str)
            .str.startswith(f"{organism_id}.")
        )

        df_org = df.loc[mask].copy()

        if len(df_org) > 0:
            result[organism_id] = df_org

    return result

#%%
# percorre TODAS as pastas e subpastas

for root, dirs, files in os.walk(path_data_processed):

    # ignorar pasta específica
    # FEAT: IGNORAR ARQUIVOS ESPECIFICOS TAMBÉM A PARTIR DE UMA ESTRUTURA DE DADOS
    if "synthesize_features" in root:
        continue

    for file in files:

        # pega apenas TSV
        if file.endswith(".tsv") and file not in arquivos_ignorados:

            file_path = os.path.join(root, file)

            print(f"\nARQUIVO ENCONTRADO:\n{file_path}")

            try:

                df = pd.read_csv(
                    file_path,
                    sep=None,
                    engine='python'
                )

                key_name = os.path.splitext(file)[0]

                # nome da chave
                if is_all_embedding_file(file):

                    dfs_por_organismo = split_embedding_by_organism(df)

                    for organism_id, df_org in dfs_por_organismo.items():

                        new_key = f"{key_name}.{organism_id}"

                        dict_df[new_key] = df_org

                        print(
                            f"Embedding ALL particionado "
                            f"para organismo {organism_id} "
                            f"-> shape {df_org.shape}"
                        )

                else:

                    dict_df[key_name] = df

                print(f"DataFrame carregado: {key_name}")
                print(f"Shape: {df.shape}")
                print(f"Colunas: {df.columns.tolist()}")

            except Exception as e:

                print(f"Erro ao carregar {file_path}")
                print(e)

#%%
seconds_end = time.time()

print(
    f"\nTempo total: "
    f"{seconds_end - seconds_ini:.2f} segundos"
)


#%%
def create_merge_key(df):

    """
    Cria chave universal de merge.
    """

    df = df.copy()

    found_key = None

    for col in possible_keys:

        if col in df.columns:

            found_key = col
            break

    if found_key is None:

        return None

    df['__MERGE_KEY__'] = (
        df[found_key]
        .astype(str)
        .str.strip()
    )

    return df

#%%
# merge principal

for k, v in dict_df.items():

    # detecta organismo diretamente no nome do arquivo
    organism_found = None

    for organism_id in dict_organismo.keys():

        if (
            k.startswith(f"{organism_id}.")
            or f".{organism_id}" in k
        ):

            organism_found = organism_id
            break

    # ignora caso não encontre organismo
    if organism_found is None:

        print(f'\nOrganismo não identificado para: {k}')
        continue

    print("\n" + "=" * 80)
    print(f"PROCESSANDO: {k}")
    print(f"ORGANISMO: {organism_found}")
    print("=" * 80)

    # cria chave universal
    v = create_merge_key(v)

    if v is None:

        print("Nenhuma chave válida encontrada.")
        continue

    print(f"Shape dataframe atual: {v.shape}")

    # primeiro dataframe
    if dict_features_merged[organism_found] is None:

        dict_features_merged[organism_found] = v.copy()

        print("DataFrame base inicializado.")

    else:

        df_base = dict_features_merged[organism_found]

        # garante chave universal
        df_base = create_merge_key(df_base)

        print(f"Shape base antes merge: {df_base.shape}")

        try:

            # remove colunas duplicadas
            duplicated_cols = [

                c for c in v.columns

                if (
                    c in df_base.columns
                    and c != '__MERGE_KEY__'
                )
            ]

            if len(duplicated_cols) > 0:

                print(
                    f"Removendo colunas duplicadas:\n"
                    f"{duplicated_cols}"
                )

                v = v.drop(columns=duplicated_cols)

            # merge universal
            merged_df = pd.merge(
                df_base,
                v,
                on='__MERGE_KEY__',
                how='inner'
            )

            dict_features_merged[
                organism_found
            ] = merged_df

            print("Merge realizado com sucesso.")
            print(f"Novo shape: {merged_df.shape}")

        except Exception as e:

            print(f"Erro no merge de {k}")
            print(e)

#%%
#adicionar classificação de essencial

path_essential = os.path.abspath(
    os.path.join(
        calculo_feature_dir,
        "../data/raw/essential"
    )
)

df_essential = pd.read_csv(
    os.path.join(
        path_essential,
        "essential_genes.csv"
    )
)

df_essential = df_essential[['Locus']]

# normaliza chave
df_essential['__MERGE_KEY__'] = (
    df_essential['Locus']
    .astype(str)
    .str.strip()
)

#%%
# adiciona coluna essential

for organism, df in dict_features_merged.items():

    if df is None:

        continue

    dict_features_merged[organism][
        'essential'
    ] = (
        df['__MERGE_KEY__']
        .isin(df_essential['__MERGE_KEY__'])
        .astype(int)
    )

#%%
# salvar arquivos finais

os.makedirs(final_path, exist_ok=True)

for k, v in dict_features_merged.items():

    if v is None:

        continue

    print("\n" + "-" * 80)
    print(f"SALVANDO: {k}")
    print("-" * 80)

    print(f"Shape final: {v.shape}")

    cols_drop = [

        '__MERGE_KEY__',
        'GeneID',
        'Protein_key',
        'preferred_name'
    ]

    existing_cols = [

        c for c in cols_drop

        if c in v.columns
    ]

    v = v.drop(
        columns=existing_cols
    )

    path_arq = os.path.join(
        final_path,
        f'{k}.synthesized_features.csv'
    )

    v.to_csv(
        path_arq,
        index=False
    )

    print(f"Arquivo salvo em:\n{path_arq}")

#%%
print("\nPROCESSAMENTO FINALIZADO.")
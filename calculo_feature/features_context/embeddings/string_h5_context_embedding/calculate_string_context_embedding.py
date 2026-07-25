#%%
import os
import sys
import pandas as pd
import h5py
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()
__SEED__ = int(os.getenv("SEED"))
__TEST_SIZE__ = float(os.getenv("TESTE_SIZE"))




project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
print(project_root)

sys.path.append(project_root)
from calculo_feature.synthesizeUtils import train_test_between_files

__X_TRAIN__, __X_TEST__, __Y_TRAIN__, __Y_TEST__ = train_test_between_files()
sys.path.remove(project_root)


calculo_feature_dir = os.getcwd()

path_data_raw = os.path.abspath(
    os.path.join(calculo_feature_dir, "../data/raw/string/embeddings/feature_context/")
)

path_data_processed = os.path.abspath(
    os.path.join(calculo_feature_dir, "../data/processed/embeddings/context/")
)

final_path = os.path.join(
    path_data_processed,
    'synthesized'
)
print(calculo_feature_dir)
print(path_data_processed)
print(path_data_raw)
print(final_path)

#%%
rows = []

for root, dirs, files in os.walk(path_data_raw):
    for file in files:
        path_arquivo = os.path.join(root, file)

        with h5py.File(path_arquivo, "r") as f:
            embeddings = f["embeddings"][:]
            proteins = [p.decode("utf-8") for p in f["proteins"][:]]

            for protein, embedding in zip(proteins, embeddings):
                rows.append([protein, *embedding])

embedding_dim = len(rows[0]) - 1

columns = (
    ["protein_id"] +
    [f"node2vec_pca_{i}" for i in range(embedding_dim)]
)

df = pd.DataFrame(rows, columns=columns)
        

df_man = df[
    df["protein_id"].apply(lambda x: x.split('.')[0]).isin({'6183'})
]

df_mus = df[
    df["protein_id"].apply(lambda x: x.split('.')[0]).isin({'10090'})
]

df_org_modelos = df[
    ~df["protein_id"].apply(lambda x: x.split('.')[0]).isin({'6183', '10090'})
]

pca_man = df_man.drop(columns=["protein_id"]).values
pca_mus = df_mus.drop(columns=["protein_id"]).values
pca_org_modelos = df_org_modelos.drop(columns=["protein_id"]).values

#Filter proteins
X_train, X_test, y_train, y_test = (__X_TRAIN__, __X_TEST__, __Y_TRAIN__, __Y_TEST__)

data_to_fit = df_org_modelos[df_org_modelos["protein_id"].isin(X_train["protein_id"])]

scaler = StandardScaler()
scaler_fit = scaler.fit(data_to_fit.drop(columns=["protein_id"]).values)

pca_org_modelos_stand = scaler_fit.transform(pca_org_modelos)
pca_man_stand = scaler_fit.transform(pca_man)
pca_mus_stand = scaler_fit.transform(pca_mus)

pca_data_fit  = scaler_fit.transform(data_to_fit.drop(columns=["protein_id"]).values) 


#
#
#   O ideal seria realizar um estudo de qual hyperparametro de N componentes.
#                       [5,10,20,30,50,0.68]
#               Aqui irei realizar com 30. 
#

df_pos_pca_org = pd.DataFrame({
    "protein_id": df_org_modelos["protein_id"].values
})

df_pos_pca_man = pd.DataFrame({
    "protein_id": df_man["protein_id"].values
})

df_pos_pca_mus = pd.DataFrame({
    "protein_id": df_mus["protein_id"].values
})

hyperparametro_pca = [5,10,20,30,50,0.68]

for pca_param in hyperparametro_pca:
    pca_calc = PCA(n_components=pca_param, random_state=__SEED__)
    pca_calc_fit = pca_calc.fit(pca_data_fit)

    pca_fin_org = pca_calc_fit.transform(pca_org_modelos_stand)
    pca_fin_man = pca_calc_fit.transform(pca_man_stand)
    pca_fin_mus = pca_calc_fit.transform(pca_mus_stand)

    #renomeando as colunas
    n_components_final = pca_fin_org.shape[1]

    colunas_pca = [
        f"node2vec_n{str(pca_param).replace('.', '_')}_pca_{i}"
        for i in range(n_components_final)
    ]

    df_org_temp = pd.DataFrame(
        pca_fin_org,
        columns=colunas_pca
    )

    df_man_temp = pd.DataFrame(
        pca_fin_man,
        columns=colunas_pca
    )

    df_mus_temp = pd.DataFrame(
        pca_fin_mus,
        columns=colunas_pca
    )

    df_pos_pca_org = pd.concat(
        [df_pos_pca_org, df_org_temp],
        axis=1
    )

    df_pos_pca_man = pd.concat(
        [df_pos_pca_man, df_man_temp],
        axis=1
    )

    df_pos_pca_mus = pd.concat(
        [df_pos_pca_mus, df_mus_temp],
        axis=1
    )

k = "node2vec"
path_pasta_model = os.path.join(path_data_processed,k)
# saves
df_pos_pca_org.to_csv(os.path.join(path_pasta_model, f"all.global.embedding.pca.{k}.tsv"), sep=' ', index=False)
df_pos_pca_man.to_csv(os.path.join(path_pasta_model, f"6183.global.embedding.pca.{k}.tsv"), sep=' ', index=False)
df_pos_pca_mus.to_csv(os.path.join(path_pasta_model, f"10090.global.embedding.pca.{k}.tsv"), sep=' ', index=False)

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import shap
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data_url = 'https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv'
data = pd.read_csv(data_url)
feature_names = data.columns[:-1]  

X_test = data.drop('Outcome', axis=1)
y_test = data['Outcome']

# Charger les modèles
scaler = joblib.load('models/scaler.pkl')
models = {
    "Logistic Regression": joblib.load('models/logistic_regression.pkl'),
    "Decision Tree": joblib.load('models/decision_tree.pkl'),
    "Random Forest": joblib.load('models/random_forest.pkl'),
    "XGBoost": joblib.load('models/xgboost.pkl'),
    "Neural Network": joblib.load('models/neural_network.pkl')
}

X_test_scaled = scaler.transform(X_test)

performance_metrics = {}

for model_name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    performance_metrics[model_name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }


def preprocess_input(data):
    data = np.array(data).reshape(1, -1)
    return scaler.transform(data)

# Fonction pour afficher les plots SHAP 
def st_shap(plot, height=None):
    """Renders the given SHAP plot in Streamlit."""
    import streamlit.components.v1 as components
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def get_user_data():
    pregnancies = st.number_input("Nombre de grossesses", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Concentration en glucose", min_value=0, max_value=200, value=120)
    blood_pressure = st.number_input("Pression artérielle diastolique (mm Hg)", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Épaisseur de la peau du triceps (mm)", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insuline sérique (mu U/ml)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("Indice de masse corporelle (poids en kg/(taille en m)^2)", min_value=0.0, max_value=100.0, value=25.0)
    dpf = st.number_input("Fonction de pedigree diabétique", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Âge (années)", min_value=0, max_value=120, value=30)
    return [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]


def predict_and_display(model_choice, user_data_scaled):
    model = models[model_choice]
    prediction = model.predict(user_data_scaled)
    prediction_proba = model.predict_proba(user_data_scaled)[:, 1]

    st.subheader(f"Prédiction : {'Diabétique' if prediction[0] == 1 else 'Non-Diabétique'}")
    st.subheader(f"Probabilité de diabète : {prediction_proba[0]:.2f}")

    if model_choice == "XGBoost":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(user_data_scaled)
        st.header("Explication de la Prédiction")
        
        shap_fig = shap.force_plot(explainer.expected_value, shap_values.values, user_data, feature_names=feature_names)
        st_shap(shap_fig, height=300)
    else:
        st.info("Les explications SHAP sont disponibles uniquement pour XGBoost")
        
    st.subheader("Performance du Modèle")
    st.write(f"**Modèle : {model_choice}**")
    st.write(f"Précision (Accuracy) : {performance_metrics[model_choice]['Accuracy']:.2f}")
    st.write(f"Rappel (Recall) : {performance_metrics[model_choice]['Recall']:.2f}")
    st.write(f"F1 Score : {performance_metrics[model_choice]['F1 Score']:.2f}")


st.set_page_config(page_title="App Prédiction de la Progression du Diabète chez des patients", layout='wide')

st.title("App Prédiction de la Progression du Diabète chez des patients")


tabs = st.tabs(["Apercu des données" , "Prédiction Individuelle", "Prédiction en Masse", "Distribution des Caractéristiques", "Matrice de Corrélation", "Box Plots", "Aide et Information"])

# Onglet Aperçu des données
with tabs[0]:
    st.header("Aperçu du Jeu de Données")
    
    st.subheader("Apercu des premières lignes")
    st.write(data.head())
    
    
    st.subheader("Description du des données")
    st.write(data.describe())

# Onglet Prédiction Individuelle
with tabs[1]:
    st.header("Prédiction Individuelle")
    with st.sidebar:
        st.header("Saisir les données médicales")
        user_data = get_user_data()
        user_data_scaled = preprocess_input(user_data)

    model_choice = st.selectbox("Choisissez le modèle", list(models.keys()))

    if st.button("Prédire"):
        predict_and_display(model_choice, user_data_scaled)

# Onglet Prédiction en Masse
with tabs[2]:
    st.header("Télécharger des données pour prédictions en masse")
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Aperçu des données téléchargées")
        st.write(df.head())

        df_scaled = scaler.transform(df)
        model = models[model_choice]
        predictions = model.predict(df_scaled)
        predictions_proba = model.predict_proba(df_scaled)[:, 1]

        df['Prediction'] = predictions
        df['Probability'] = predictions_proba

        st.write("Résultats des prédictions")
        st.write(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger les résultats", csv, "predictions.csv", "text/csv", key='download-csv')

# Onglet Distribution des Caractéristiques
with tabs[3]:
    st.header("Distribution des Caractéristiques")
    feature_choice = st.selectbox("Choisir une caractéristique", data.columns[:-1])
    fig = px.histogram(data, x=feature_choice, title=f"Distribution de {feature_choice}")
    st.plotly_chart(fig)

# Onglet Matrice de Corrélation
with tabs[4]:
    st.header("Matrice de Corrélation")
    correlation_matrix = data.corr()

    fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

# Onglet Box Plots
with tabs[5]:
    st.header("Box Plots des Caractéristiques")
    feature_choice_box = st.selectbox("Choisir une caractéristique pour le Box Plot", data.columns[:-1])

    fig_box, ax_box = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=data[feature_choice_box], ax=ax_box)
    ax_box.set_title(f"Box Plot de {feature_choice_box}")
    st.pyplot(fig_box)
    
    
# Onglet Aide et Information
with tabs[6]:
    st.header("Aide et Information")

    st.subheader("Comment utiliser l'application")
    st.markdown("""
    1. **Prédiction Individuelle** :
        - Saisissez les données médicales d'un patient dans la barre latérale.
        - Sélectionnez le modèle de prédiction souhaité.
        - Cliquez sur le bouton "Prédire" pour obtenir le résultat.
    2. **Prédiction en Masse** :
        - Téléchargez un fichier CSV contenant les données des patients.
        - L'application affichera un apercu des données.
        - Cliquer sur "Télécharger les résultats" pour obtenir les prédictions en fichier CSV.
    3. **Distribution des Caractéristiques** :
        - Sélectionnez une caractéristique dans la liste déroulante.
        - Un histogramme affichera la distribution de la caractéristique sélectionnée.
    4. **Matrice de Corrélation** :
        - Affiche la matrice de corrélation entre les différentes caractéristiques des données.
    5. **Box Plots** :
        - Sélectionnez une caractéristique pour visualiser sa distribution à l'aide d'un box plot.
    """)

    st.subheader("Comment interpréter les résultats")
    st.markdown("""
    - **Prédiction** : La prédiction indique si le patient est considéré comme diabétique (1) ou non-diabétique (0).
    - **Probabilité de diabète** : Cette valeur indique la probabilité que le patient soit diabétique.
    - **Graphiques SHAP (pour XGBoost)** : Les graphiques SHAP montrent l'impact de chaque caractéristique sur la prédiction. Les valeurs positives augmentent la probabilité de diabète, tandis que les valeurs négatives la diminuent.
    """)

    st.subheader("Informations sur les caractéristiques des données")
    st.markdown("""
    - **Nombre de grossesses** : Le nombre de fois que la patiente a été enceinte.
    - **Concentration en glucose** : Niveau de glucose dans le sang.
    - **Pression artérielle diastolique** : Pression artérielle diastolique (mm Hg).
    - **Épaisseur de la peau du triceps** : Épaisseur de la peau du triceps (mm).
    - **Insuline sérique** : Niveau d'insuline sérique (mu U/ml).
    - **Indice de masse corporelle (IMC)** : Poids en kg/(taille en m)^2.
    - **Fonction de pedigree diabétique** : Score de pedigree diabétique, représentant l'historique familial de diabète.
    - **Âge** : Âge du patient en années.
    """)
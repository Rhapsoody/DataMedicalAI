# Prédiction de la Progression du Diabète

Projet d'analyse des données sur le diabète et utilisation de modèles pour la prédiction

## Contenu 

- **Notebook d'Analyse et de Modélisation** : Exploration et préparation des données, entraînement des modèles 

- **Application Streamlit** : Interface utilisateur streamlit pour interagir avec le modèle

## Installation

1. Clonez le dépôt :
    ```bash
    git clone https://github.com/Rhapsoody/DataMedicalAI.git
    cd <nom_du_dépôt>
    ```

2. Créez et activez un environnement virtuel :
    ```bash
    python -m venv venv
    source venv/bin/activate  
    ```

3. Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

1. Lancez l'application Streamlit :
    ```bash
    streamlit run app.py
    ```

2. Vous aurez accès à :
    - **Prédiction Individuelle** : Saisissez les données médicales d'un patient pour obtenir une prédiction de la probabilité de diabète
    - **Prédiction en Masse** : Téléchargez un fichier CSV contenant les données de plusieurs patients pour obtenir des prédictions en masse
    - **Distribution des Caractéristiques** : Visualisez les distributions des caractéristiques médicales
    - **Matrice de Corrélation** : Affichez la matrice de corrélation entre les différentes caractéristiques
    - **Box Plots** : Visualisez la distribution des caractéristiques à l'aide de box plots
    - **Aperçu des Données** : Consultez un aperçu des premières lignes et des statistiques descriptives du jeu de données

**Créé par** : AGONGLO Shalom
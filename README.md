# Detection et Classification Automatique de Maladies des Plantes

Projet organise pour le sujet "Detection et classification automatique de maladies des plantes" a partir du dataset PlantVillage.

## Choix de travail

Le sujet impose un travail individuel sur 4 classes. Ce projet est configure par defaut sur 4 classes de tomate pour eviter qu'un modele apprenne surtout l'espece au lieu de la maladie :

- `Tomato___healthy`
- `Tomato___Early_blight`
- `Tomato___Late_blight`
- `Tomato___Bacterial_spot`

Ce choix est modifiable dans [configs/default.yaml](/C:/Users/ASUS/Documents/IGL4/ingenieurie_d'image/Traitement-d'image/configs/default.yaml).

## Structure

- `configs/` : configuration du projet
- `docs/` : canevas de rapport et livrables
- `scripts/` : scripts d'execution
- `src/plant_disease/` : code source

## Dataset

1. Telecharger le dataset Kaggle `emmarex/plantdisease`.
2. Extraire les images dans `data/raw/PlantVillage/`.

Le projet attend une structure du type :

```text
data/raw/PlantVillage/
  Tomato___healthy/
  Tomato___Early_blight/
  Tomato___Late_blight/
  Tomato___Bacterial_spot/
```

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline de travail

1. Indexer le dataset et creer les splits :

```powershell
python scripts/build_index.py --config configs/default.yaml
```

2. Generer des exemples de pretraitement / segmentation :

```powershell
python scripts/generate_examples.py --config configs/default.yaml
```

3. Entrainer le modele classique sur des caracteristiques extraites :

```powershell
python scripts/train_classical.py --config configs/default.yaml
```

4. Entrainer le modele deep learning :

```powershell
python scripts/train_deep.py --config configs/default.yaml
```

5. Lancer tout le pipeline :

```powershell
python scripts/run_full_pipeline.py --config configs/default.yaml
```

## Ce que couvre le projet

- Pretraitement image : redimensionnement, denoising leger, CLAHE, normalisation
- Segmentation / contours : masque HSV de la feuille, morphologie, Canny
- Extraction de caracteristiques : couleur, texture, forme, contours
- Modele classique : `StandardScaler + SVC` et `RandomForest`
- Modele deep learning : CNN simple en PyTorch, avec option transfer learning si `torchvision` est disponible
- Evaluation : accuracy, precision, recall, F1, matrice de confusion
- Livrables : exemples visuels, fichiers de metriques, canevas de rapport

## Sorties attendues

Apres execution, les artefacts sont ranges dans :

- `artifacts/metadata/`
- `artifacts/examples/`
- `artifacts/classical_ml/`
- `artifacts/deep_learning/`

## Remarques

- Le sujet demande une comparaison ML classique vs DL : ce projet est structure pour produire cette comparaison proprement.
- Les idees recurrentes observees dans les notebooks Kaggle sur PlantVillage sont integrees dans la structure : sous-ensemble de classes, augmentation, transfer learning, confusion matrix, comparaison des pipelines.
- Si vous voulez changer les 4 classes, modifiez seulement `configs/default.yaml`.


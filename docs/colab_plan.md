# Plan Exact Pour Travailler Avec Colab

## 1. Choisir les 4 classes

Garde les 4 classes de tomate deja configurees dans le projet :

- `Tomato___healthy`
- `Tomato___Early_blight`
- `Tomato___Late_blight`
- `Tomato___Bacterial_spot`

Ne change pas les noms entre Colab et le projet local.

## 2. Sur Colab

1. Ouvrir un notebook Colab
2. Aller dans `Runtime > Change runtime type`
3. Choisir `GPU`
4. Copier le contenu de `notebooks/colab_pretrained_training.py` cellule par cellule
5. Ajouter ton fichier `kaggle.json` quand Colab le demande
6. Lancer toutes les cellules
7. Recuperer les fichiers exportes :
   - `best_model.pt`
   - `class_names.json`
   - `training_summary.json`
   - `confusion_matrix.png`

## 3. Ou mettre les fichiers ensuite dans ton projet

Copier les sorties Colab dans :

```text
artifacts/deep_learning/
```

Au minimum :

```text
artifacts/deep_learning/best_model.pt
artifacts/deep_learning/class_names.json
artifacts/deep_learning/training_summary.json
artifacts/deep_learning/confusion_matrix.png
```

## 4. Ce que tu lances en local

Quand le dataset local est present :

```powershell
python scripts/build_index.py --config configs/default.yaml
python scripts/generate_examples.py --config configs/default.yaml
python scripts/train_classical.py --config configs/default.yaml
```

La partie deep learning sera deja entrainee sur Colab.

## 5. Ce que tu dois mettre dans le rapport

- pourquoi tu as choisi 4 classes de tomate
- le pretraitement applique
- la segmentation HSV + contours
- les caracteristiques extraites pour SVM / RandomForest
- le modele pretrained choisi pour Colab
- les hyperparametres d'entrainement
- les metriques finales
- la comparaison entre ML classique et Deep Learning

## 6. Modele recommande

Pour ce sujet, prends `EfficientNet-B0` pretrained.

Pourquoi :

- leger
- rapide sur Colab GPU
- tres courant dans les notebooks Kaggle sur PlantVillage
- generalement meilleur qu'un petit CNN from scratch sur peu d'effort


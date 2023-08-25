# Perceptron Classifier

Ce code Python est un exemple d'utilisation du perceptron, une technique de classification de base en apprentissage automatique. Le code utilise scikit-learn, une bibliothèque Python populaire pour l'apprentissage automatique, pour charger les données, standardiser les caractéristiques, réduire la dimensionnalité avec PCA, entraîner un modèle de perceptron et faire des prédictions.

## Installation

Ce code nécessite Python 3 et les bibliothèques suivantes :

* numpy
* matplotlib
* scikit-learn

Vous pouvez installer les bibliothèques en utilisant pip :

pip install numpy matplotlib scikit-learn

## Utilisation

Vous pouvez exécuter le code en utilisant la commande suivante :

python perceptron_classifier.py

Le code charge les données à partir de l'ensemble de données iris, divise les données en ensembles d'entraînement et de test, standardise les caractéristiques, réduit la dimensionnalité avec PCA, entraîne un modèle de perceptron et fait des prédictions. La précision du modèle est imprimée à la sortie.

Vous pouvez également visualiser les performances du modèle en appelant la méthode `plot_performance()` de l'objet `PerceptronClassifier`. Cette méthode trace le nombre de mises à jour du modèle par epoch.

## Remarques

Ce code est fourni à titre d'exemple et peut être modifié pour s'adapter à vos besoins. Le code peut être exécuté sur d'autres ensembles de données en modifiant les appels aux fonctions de chargement de données et de séparation en ensembles d'entraînement et de test.

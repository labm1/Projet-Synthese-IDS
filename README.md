## Usage of Artificial Intelligence models for the detection of intrusions in a network
This project was developed as part of the university course INF4173 Final Project at UQO in Winter 2025.

The goal of this project is to realise a prototype of a Intrusion Detection System (IDS) using pre-existing Artificial Intelligence (AI) models, and more precisely machine learning models. The performances of these models are then compared to find which one is the best for intrusion detection.

Documentation for this project is in french.

# L’utilisation de modèles d’intelligence artificielle pour la détection des intrusions dans les réseaux
Le but du projet est de réaliser un prototype de système de détection des intrusions (IDS) avec des modèles préexistants d’Intelligence Artificielle (IA), plus précisément d’apprentissage automatique. L’objectif est de comparer leurs performances pour déterminer le modèle le plus efficace pour la détection d’attaques.

L’intelligence artificielle est une technologie puissante, et a une grande utilité dans le domaine de la cybersécurité, en permettant de reconnaitre une cyberattaque, et l’isoler de son environnement pour ne pas cause de dégâts, ce qui en fait un outil bien adapté pour la détection d’intrusions.

Pour entrainer les données, la base de données publique NSL-KDD est utilisée. Elle contient des données de trafic réseau, incluant des données normales et d’attaques. Il y a quatre types d’attaques présentes : _DoS_, _Probe_, _R2L_ et _U2R_.

Pour réaliser le prototype, le langage Python est utilisé, avec la bibliothèque sklearn, où se trouvent les différents modèles de classification qui sont utilisés et comparés. Ces modèles sont : _Decision Tree_, _Random Forest_, _K-Nearest Neighbors_, _Logistic Regression_, _Naïve Bayes_ et _Support Vector Machines_. Ils sont évalués et comparés selon les métriques d’exactitude, précision, rappel et F-mesure. Le graphe de la courbe ROC est également utilisé pour comparer les modèles dans leur précision.

Après la comparaison des modèles, il est clair que c’est _Random Forest_ qui est le meilleur pour prédire les attaques en général, ainsi que pour les attaques spécifiques DoS, Probe et R2L, suivi de près par _Decision Tree_. Pour l’attaque U2R, il semble que le modèle le plus efficace soit _Logistic Regression_, mais ce n’est pas possible de dire avec certitude, puisque ce type d’attaque n’apparait que très peu dans la base de données, ce qui empêche les modèles d’être bien entrainés sur ce type d’attaque. Le pire modèle est _Naïve Bayes_ pour tous les types d’attaques.

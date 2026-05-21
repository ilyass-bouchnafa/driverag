"""
evaluation/test_questions.py
----------------------------
Collection of test questions and ground-truth answers used by
the RAGAS evaluation. These items are written in French and are
used as sample evaluation data. Keep the content language as-is
unless you want to provide an English dataset.
"""

TEST_QUESTIONS = [
    "À quoi sert le privilège RESOURCE accordé à un utilisateur Oracle ?",
    "Quel est l'avantage d'utiliser NUMBER(10,2) pour le champ prix de la table Produit en PL/SQL ?",
    "Que fait l'instruction EXIT WHEN c_prod%NOTFOUND dans une boucle de curseur PL/SQL ?",
    "Qu'est-ce qu'un BLOB en base de données ?",
    "Selon l'article d'Alan Turing en 1950, quel est le but principal de l'intelligence artificielle ?",
    "Quelle est la principale différence de capacité entre une IA faible et une IA forte ?",
    "Quel est le principe de l'algorithme K Plus Proches Voisins (KNN) pour classer une nouvelle donnée ?",
    "Quel est le rôle de la fonction noyau dans les Machines à Vecteur de Support (SVM) ?",
    "En traitement d'image, à quoi correspond l'échantillonnage numérique ?",
    "Quelle est la différence fondamentale de structure entre une image vectorielle et une image matricielle ?",
    "Quel est le but et le résultat principal d'une égalisation d'histogramme sur une image ?",
    "Quelles sont les caractéristiques principales du filtre médian en analyse d'image ?",
    "Quelles solutions existent pour traiter les effets de bord lors de l'application d'un filtre de convolution ?",
    "De quelles couleurs de base est composé le système de représentation RGB ?",
    "Qu'est-ce que le système de gestion de fichiers (SGF) et quel est son rôle global ?",
    "Que contient le super bloc d'une partition de disque sous Unix ?",
    "Dans l'en-tête d'un fichier binaire exécutable, à quoi sert le 'Nombre magique' ?",
    "Quelles sont les informations contenues dans un i-nœud ?",
    "Que se passe-t-il pour un lien symbolique lorsque le fichier original est supprimé ?",
    "Comment fonctionne l'allocation physique de fichiers par liste chaînée de blocs ?"
]

GROUND_TRUTHS = [
    "Le privilège RESOURCE permet à l'utilisateur de créer des tables, des séquences, des procédures et d'autres objets dans la base de données[cite: 19].",
    "Utiliser NUMBER(10,2) permet de stocker un prix avec un haut niveau de précision, en incluant deux décimales[cite: 29, 30].",
    "Cette instruction permet d'arrêter la boucle de lecture lorsque toutes les lignes retournées par le curseur ont été parcourues[cite: 201].",
    "Un BLOB (Binary Large Object) est un type de données utilisé pour stocker des fichiers binaires de grande taille, tels que des images, des vidéos ou des documents PDF[cite: 220].",
    "Le but de l'IA est de concevoir des systèmes capables de reproduire le comportement humain dans ses activités de raisonnement[cite: 268, 280].",
    "Une IA faible résout des problèmes en détectant des motifs répétitifs, tandis qu'une IA forte (qui n'existe pas encore) serait capable de penser, raisonner et manipuler des concepts abstraits au même niveau qu'un être humain[cite: 422].",
    "L'algorithme KNN attribue un nouveau point à la catégorie la plus présente parmi ses K voisins les plus proches, après avoir calculé la distance qui les sépare[cite: 461, 469].",
    "La fonction noyau permet de projeter les données d'entrée dans un espace de très grande dimension lorsque celles-ci ne sont pas linéairement séparables[cite: 493, 494].",
    "L'échantillonnage correspond à la discrétisation de l'espace 2D de l'image, ce qui définit le nombre exact de points (pixels) qui pourront être coloriés[cite: 620, 661].",
    "Une image vectorielle est représentée par des formes géométriques simples permettant un redimensionnement sans perte, contrairement à une image matricielle qui est représentée par une matrice de pixels (Bitmap)[cite: 626, 628].",
    "L'égalisation d'histogramme équilibre la distribution des pixels, ce qui a pour résultat d'augmenter globalement le contraste de l'image[cite: 677, 678].",
    "Le filtre médian est un filtre non linéaire qui remplace la valeur d'un pixel par la médiane de ses voisins, ce qui supprime le bruit de type 'Poivre et Sel' tout en préservant l'information de contour[cite: 732, 733].",
    "Pour traiter les effets de bord, on peut utiliser le 'zero padding' (considérer le voisinage hors image comme des valeurs nulles) ou la 'duplication' (attribuer la valeur du pixel de l'image le plus proche)[cite: 708, 709, 711].",
    "Le système RGB est composé de trois couleurs primaires : le Rouge, le Vert et le Bleu[cite: 647, 650].",
    "Le SGF est la partie du système d'exploitation chargée de gérer les fichiers, incluant leur création, suppression, les accès en lecture/écriture, le partage, la protection et l'allocation de l'espace disque[cite: 771, 772].",
    "Le super bloc contient les paramètres clés du système de fichiers, notamment la taille des blocs, le nombre de blocs libres et la taille de la table des i-nœuds[cite: 886].",
    "Le 'Nombre magique' est une valeur particulière qui indique au système d'exploitation que le fichier respecte un format l'identifiant comme exécutable[cite: 847].",
    "Un i-nœud contient des informations de localisation (comme 13 adresses disques directes et indirectes) ainsi que des attributs généraux tels que le type de fichier, le propriétaire, la taille, les dates et les droits d'accès[cite: 956, 959].",
    "Lors de la suppression du fichier original, le lien symbolique devient inutilisable car il pointe vers un chemin d'accès qui n'existe plus[cite: 884].",
    "Le fichier est conservé sous forme d'une liste de blocs dispersés sur le disque ; il suffit de garder l'adresse du premier bloc, car chaque bloc de la liste contient un pointeur vers le bloc suivant[cite: 895]."
]
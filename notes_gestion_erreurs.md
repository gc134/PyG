# Mise en place d'un système de gestion des erreurs pour PyG

## idées

* on se contente d'une gestion des erreurs basiques
* qd ia une erreur on ne cherche pas à l'analyser finement pour donner des pistes précises à l'utilisateur, mais on se contente d'un message générique du genre "il y a un problème dans le contenu de votre tableau. Veuillez respecter les règles précisées dans la notice". A l'utilisateur
* pour éviter les plantages, bugs, ia un mot-clé très utile en python c'est "pass"
* en python toutes les erreurs sont traitées sous forme d'exceptions = objets contenant des infos sur le contexte, la nature de l'erreur
* phase typique pour gestion des exceptions en python :
  * test d'une instruction : try
  * levée, détection de l'erreur : try
  * capture de l'erreur : except
  * traitement de l'erreur (solution envisagée pour éviter le plantage) : bloc de except
  * alternative s'il n'y a pas d'erreur dans le bloc de test : else
  * commande systématique : finally
* vérouiller certains boutons si les objets qui'ils traitent ne sont pas encore créé :
  * ex : page2, verrouiller le bouton d'export tant que le calcul des % n'est pas fait

## liste des étapes problématiques dans l'utilisation de l'interface

### page1

* import des données
  * nom du fichier : caract spéciaux
  * format du fichier
  * taille du fichier
  * contenu : présence de caractères spéciaux qque part
* pb sur les calculs de QC
  * lié au contenu du tableau
  * si erreur sur le QC, rappeler les règles sur contenu du tableau
* ordre des modalités
  * saisie des modalités triées sur une même ligne
  * certaines modalités n'existent pas dans la colonne groupe = saisie de modalités inexistantes
  * faire un QC de cette saisie à ce niveau, parce que c'est utilisé bcp plus loin aux étapes boxplot, et en cas de pb on aura du mal à ces étapes éloignées de faire le lien avec une mauvaise saisie des modalités
  * si rien n'est saisie, prendre l'ordre alphab

### page2

* calcul des pourcentages
  * dépend de la nature des données de comptage
  * nécessite une colonne "total"
  * necessité des nombres ailleurs
  * NA?
* contraindre à lancer le calcul des % avant de passer à la page suivante : message
* vérifier la structure-nature du tableau de % avant de passer aux courbes de germ

### page3

* calcul des courbes de germ indiv et groupées
  * lié à des pb au niveau du tableau des %
  * message demandant de vérifier la nature du tableau de % en rappelant les règles pour ce tableau
* pas de pb attendu au niveau des fenêtres matplotlib, l'outil ayant ses propres système d'exceptions

### page4

* saisie des valeurs initiales pour l'ajustement
  * valeurs hos gamme ou n'ayant pas de sens
  * fixer des limites aux paramètres
  * avertissements si valeurs saisies aberrantes
* pb d'ajustement, de calculs des paramètres des modèles
  * données globalement insuffisantes : prévenir l'utilisateur que le nb de points temporels est peut-être insuffisant pour extraire les paramètres (nb de points insuffisants) et qu'il doit revoir son PE (plan d'exp)
  * données aberrantes au sens de l'évolution temporelle (pic ou vallée dans la courbe)
  * données insuffisantes pour certains lots : courbes plates par exemple
  * quels comportements de la fonction d'ajustement en cas d'impossibilité de calcul : générer des NA ou NaN pour certains ech°
  * inviter à corriger les valeurs initiales?
* pb d'affichage au niveau courbes d'ajustement à cause de données insuffisantes :
  * mêmes origines que pour les paramètres d'ajustement
  * pb qui se répercute au niveau de l'image de remplissage et de la fenêtre matplotlib
  * comment gérer ces problèmes d'impossibilité de générer les graphes : avertissement sur l'insuffisance des données
* pb de calcul des paramètres de germ indiv et groupés
  * mêmes origines que précedemment
  * comment gérer les valeurs manquantes pour certains lots : générer des NA?
  * quelles influences pour calcul des moyennes
* mettre en place un verrou, ou un avertissement sur le passage dans la partie stat par rapport aux pb éventuels au niveau paramètres de germination : ce sont les données du tableau des paramètres de germ indiv qui sont dès lors utilisées

### page5

* les données sont elles suffisamment de qualité pour calculer des boxplots?
* si c'est non ce n'est plus la peine de continuer plus loin, les stats n'ont plus de sens
* invitation à relancer l'expérience pour avoir plus de données permettant de faire des stat
* éliminer le plantage qd on lance un boxplot sur la sélection "Choix d'un paramètre"
* pas d'autres erreurs envisagées
* prévoir l'erreur d'un ordre des modalités, saisie en page1, erronné

### page6

* stat
* les données sont-elles suffisantes pour faire toutes les stat
* corriger le bug si "Choix d'un paramètre" est pris
* si c'est passer pour les boxplot ça devrait passer pour les stat
* même chose pour le bilan des p-values
* comment faire si un des paramètres pose pb = les autres sont suffisants pour des stat, mais impossibilté de calcul pour ce paramètre
* générer des NA en cas d'impossibilité, mais les outils stat utilisés doivent avoir leurs propres gestion des erreurs
* warning sur l'impossibilté de faire des stats sur un paramètre après avoir essayer
* gérer erreur d'export si tableView est vide

### page7

* gérer bug si "Choix d'un paramètre" est sélectionné
* comme le boxplot de Tukey reprend l'ensemble des p-values de comp multiples calculées entre les paires de groupe, il faut que le vecteur utilisé pour le tri des modalités soit complet, cad contienne toutes les modalités
* si ia des erreurs, ça viendra plus sûrement des résultats de la partie stat
* en cas d'impossibilté de sortir un boxplot de Tukey, avertir l'utilisateur que le calcul est impossible sur ses données. Ne pas ouvrir de fenêtre matplotlib et ne rien chercher à afficher dans le zone de remplissage
* si ia un test multiple qui ne s'est pas fait correctement, comment ça apparaitra sur le boxplot de Tukey
* gérer le fait qu'il y ait dans l'analyse un gd nb de groupes (+ de 5) qui nécessite un gd nb de p-values multiples. Comment ce sera gérer par matplotlib et quel affichage miniature
* attention aux erreurs liées à un vecteur d'ordre de modalités incorrect
* afficher warning invitant à corriger ce vecteur s'il n'est pas bon et  éviter la production des graphes

import random
import math
import numpy as np

def calculer_distance_totale(solution, matrice_distances):
    """Calcule la distance totale d'un parcours"""
    distance_totale = 0
    for i in range(len(solution) - 1):
        distance_totale += matrice_distances[solution[i]][solution[i + 1]]
    distance_totale += matrice_distances[solution[-1]][solution[0]]
    return distance_totale

def generer_population_initial(taille_population, nombre_villes):
    """Génère une population initiale aléatoire"""
    population = []
    for _ in range(taille_population):
        individu = list(range(nombre_villes))
        random.shuffle(individu)
        population.append(individu)
    return population

def evaluer_population(population, matrice_distances):
    """Évalue la fitness de chaque individu (inverse de la distance)"""
    fitness = []
    for individu in population:
        distance = calculer_distance_totale(individu, matrice_distances)
        # La fitness est l'inverse de la distance (plus la distance est petite, meilleure est la fitness)
        fitness.append(1 / distance)
    return fitness

def selection_par_rang(population, fitness, taux_selection=0.5):
    """
    Sélection par rang : les individus sont classés par fitness
    et sélectionnés selon leur rang plutôt que leur fitness absolue
    """
    # Trier la population par fitness (décroissant)
    population_triee = [ind for _, ind in sorted(zip(fitness, population), reverse=True)]
    fitness_triee = sorted(fitness, reverse=True)
    
    n_selection = int(len(population) * taux_selection)
    
    # Attribution des probabilités basées sur le rang
    rangs = list(range(1, len(population) + 1))
    # Probabilité inversement proportionnelle au rang (meilleur rang = plus haute probabilité)
    probabilites = [1/r for r in rangs]
    somme_probabilites = sum(probabilites)
    probabilites = [p/somme_probabilites for p in probabilites]
    
    # Sélection des parents
    parents = []
    for _ in range(n_selection):
        # Sélection proportionnelle au rang
        choix = random.choices(population_triee, weights=probabilites)[0]
        parents.append(choix.copy())
    
    return parents

def croisement_OX(parent1, parent2):
    """
    Croisement Order Crossover (OX) pour le TSP
    Préserve l'ordre relatif des villes
    """
    taille = len(parent1)
    enfant = [-1] * taille
    
    # Choisir deux points de coupure aléatoires
    point1, point2 = sorted(random.sample(range(taille), 2))
    
    # Copier le segment entre point1 et point2 du parent1 à l'enfant
    for i in range(point1, point2 + 1):
        enfant[i] = parent1[i]
    
    # Remplir le reste avec les villes du parent2 dans l'ordre
    position = (point2 + 1) % taille
    for ville in parent2:
        if ville not in enfant:
            enfant[position] = ville
            position = (position + 1) % taille
    
    return enfant

def mutation_echange(individu, taux_mutation=0.1):
    """Mutation par échange de deux villes"""
    if random.random() < taux_mutation:
        i, j = random.sample(range(len(individu)), 2)
        individu[i], individu[j] = individu[j], individu[i]
    return individu

def algorithme_genetique(matrice_distances, taille_population=100, generations=1000, 
                        taux_selection=0.5, taux_mutation=0.1, elite_size=2):
    """
    Implémentation de l'algorithme génétique avec sélection par rang pour le TSP
    
    Args:
        matrice_distances: Matrice des distances entre villes
        taille_population: Taille de la population
        generations: Nombre de générations
        taux_selection: Pourcentage d'individus sélectionnés pour la reproduction
        taux_mutation: Probabilité de mutation
        elite_size: Nombre de meilleurs individus conservés directement
    """
    nombre_villes = len(matrice_distances)
    
    # Génération de la population initiale
    population = generer_population_initial(taille_population, nombre_villes)
    
    # Évaluation initiale
    fitness = evaluer_population(population, matrice_distances)
    
    meilleure_solution = population[np.argmax(fitness)].copy()
    meilleure_distance = calculer_distance_totale(meilleure_solution, matrice_distances)
    
    historique = []
    
    print("Démarrage de l'Algorithme Génétique...")
    print(f"Taille population: {taille_population}")
    print(f"Nombre de générations: {generations}")
    print(f"Solution initiale: {meilleure_solution} (distance: {meilleure_distance})")
    
    for generation in range(generations):
        # Sélection par rang
        parents = selection_par_rang(population, fitness, taux_selection)
        
        # Reproduction (croisement)
        nouvelle_generation = []
        
        # Élitisme : conserver les meilleurs individus
        indices_tries = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)
        for i in range(elite_size):
            nouvelle_generation.append(population[indices_tries[i]].copy())
        
        # Remplir le reste de la population avec des enfants
        while len(nouvelle_generation) < taille_population:
            parent1, parent2 = random.sample(parents, 2)
            enfant = croisement_OX(parent1, parent2)
            enfant = mutation_echange(enfant, taux_mutation)
            nouvelle_generation.append(enfant)
        
        population = nouvelle_generation
        fitness = evaluer_population(population, matrice_distances)
        
        # Mettre à jour la meilleure solution
        meilleur_fitness_idx = np.argmax(fitness)
        distance_actuelle = 1 / fitness[meilleur_fitness_idx]
        
        if distance_actuelle < meilleure_distance:
            meilleure_solution = population[meilleur_fitness_idx].copy()
            meilleure_distance = distance_actuelle
        
        # Statistiques
        distance_moyenne = 1 / (sum(fitness) / len(fitness))
        meilleur_fitness = max(fitness)
        
        if generation % 100 == 0 or generation == generations - 1:
            print(f"Génération {generation}: Meilleure={meilleure_distance:.2f}, "
                  f"Moyenne={distance_moyenne:.2f}, Fitness_max={meilleur_fitness:.6f}")
        
        historique.append({
            'generation': generation,
            'meilleure_distance': meilleure_distance,
            'distance_moyenne': distance_moyenne,
            'meilleur_fitness': meilleur_fitness
        })
    
    print("Optimisation terminée!")
    return meilleure_solution, meilleure_distance, historique

# Matrice de distances (identique à votre code original)
matrice_distances = [
    [0, 2, 2, 7, 15, 2, 5, 7, 6, 5],
    [2, 0, 10, 4, 7, 3, 7, 15, 8, 2],
    [2, 10, 0, 1, 4, 3, 3, 4, 2, 3],
    [7, 4, 1, 0, 2, 15, 7, 7, 5, 4],
    [7, 10, 4, 2, 0, 7, 3, 2, 2, 7],
    [2, 3, 3, 7, 7, 0, 1, 7, 2, 10],
    [5, 7, 3, 7, 3, 1, 0, 2, 1, 3],
    [7, 7, 4, 7, 2, 7, 2, 0, 1, 10],
    [6, 8, 2, 5, 2, 2, 1, 1, 0, 15],
    [5, 2, 3, 4, 7, 10, 3, 10, 15, 0]
]

# Exécution de l'Algorithme Génétique
print("=== ALGORITHME GÉNÉTIQUE - SÉLECTION PAR RANG ===")
meilleure_solution, meilleure_distance, historique = algorithme_genetique(
    matrice_distances,
    taille_population=50,
    generations=1000,
    taux_selection=0.6,
    taux_mutation=0.15,
    elite_size=2
)

print("\n=== RÉSULTATS FINAUX ===")
print(f"Meilleure solution trouvée: {meilleure_solution}")
print(f"Distance minimale: {meilleure_distance}")

# Comparaison des performances
print("\n" + "="*60)
print("ANALYSE DES PERFORMANCES")
print("="*60)

if len(historique) > 0:
    print(f"Distance initiale: {historique[0]['meilleure_distance']:.2f}")
    print(f"Distance finale: {historique[-1]['meilleure_distance']:.2f}")
    amelioration = ((historique[0]['meilleure_distance'] - meilleure_distance) / historique[0]['meilleure_distance']) * 100
    print(f"Amélioration: {amelioration:.1f}%")

# Version avec affichage détaillé
def algorithme_genetique_detaille(matrice_distances, taille_population=100, generations=1000, 
                                 taux_selection=0.5, taux_mutation=0.1, elite_size=2):
    """Version détaillée avec plus de statistiques"""
    nombre_villes = len(matrice_distances)
    
    population = generer_population_initial(taille_population, nombre_villes)
    fitness = evaluer_population(population, matrice_distances)
    
    meilleure_solution = population[np.argmax(fitness)].copy()
    meilleure_distance = calculer_distance_totale(meilleure_solution, matrice_distances)
    
    print("Démarrage détaillé de l'Algorithme Génétique...")
    print(f"Population: {taille_population}, Générations: {generations}")
    print(f"Sélection: {taux_selection*100}%, Mutation: {taux_mutation*100}%, Élite: {elite_size}")
    
    for generation in range(generations):
        # Sélection par rang
        parents = selection_par_rang(population, fitness, taux_selection)
        
        # Nouvelle génération avec élitisme
        nouvelle_generation = []
        
        indices_tries = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)
        for i in range(elite_size):
            nouvelle_generation.append(population[indices_tries[i]].copy())
        
        # Reproduction
        while len(nouvelle_generation) < taille_population:
            parent1, parent2 = random.sample(parents, 2)
            enfant = croisement_OX(parent1, parent2)
            enfant = mutation_echange(enfant, taux_mutation)
            nouvelle_generation.append(enfant)
        
        population = nouvelle_generation
        fitness = evaluer_population(population, matrice_distances)
        
        # Mise à jour de la meilleure solution
        meilleur_fitness_idx = np.argmax(fitness)
        distance_actuelle = 1 / fitness[meilleur_fitness_idx]
        
        if distance_actuelle < meilleure_distance:
            meilleure_solution = population[meilleur_fitness_idx].copy()
            meilleure_distance = distance_actuelle
        
        # Affichage détaillé toutes les 100 générations
        if generation % 100 == 0:
            distance_moyenne = 1 / (sum(fitness) / len(fitness))
            diversite = len(set(tuple(ind) for ind in population)) / taille_population * 100
            
            print(f"Génération {generation:4d}: "
                  f"Meilleure={meilleure_distance:6.2f}, "
                  f"Moyenne={distance_moyenne:6.2f}, "
                  f"Diversité={diversite:5.1f}%")
    
    return meilleure_solution, meilleure_distance

# Exécution de la version détaillée
print("\n" + "="*60)
print("EXÉCUTION DÉTAILLÉE")
print("="*60)

meilleure_solution_det, meilleure_distance_det = algorithme_genetique_detaille(
    matrice_distances,
    taille_population=50,
    generations=1000,
    taux_selection=0.6,
    taux_mutation=0.15,
    elite_size=2
)

print(f"\nMeilleure solution: {meilleure_solution_det}")
print(f"Distance: {meilleure_distance_det}")
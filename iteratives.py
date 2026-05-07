import math
import numpy as np
import matplotlib.pyplot as plt
from matrices import choisir_norme, is_DDS, is_positive_definite, is_symmetric, rayon_spectral ,saisir_matrice, saisir_vecteur



def jacobi(A, b, x0, epsilon, max_iter, type_norme="inf", p_norm=None):
    D=np.diag(np.diag(A))
    E=-np.tril(A,-1)
    F=-np.triu(A,1)

    D_inv=np.diag(1/np.diag(D))
    Js=D_inv @ (E+F)
    c=D_inv @ b

    M_norm = choisir_norme(Js, type_norme, p_norm)
    p=rayon_spectral(Js)

    historique_erreurs = []
    historique_X = []
    X = x0.copy()

    convergence = False
    nb_iterations = 0

    for i in range(max_iter):
       X_new = Js @ X + c
       historique_X.append(X_new.copy())

       diff = np.linalg.norm(X_new - X)

       if M_norm < 1:
          err_estime = diff * (M_norm / (1 - M_norm))
          historique_erreurs.append(err_estime)
          critere = err_estime
       else:
          err_estime = diff
          critere = diff

       if critere < epsilon:
            convergence = True
            nb_iterations = i + 1
            break

       X = X_new
    if not convergence:
       nb_iterations = max_iter

    solution = historique_X[-1]

    
     

    return convergence, solution, historique_X, historique_erreurs, p, Js, M_norm



def gauss_seidel(A, b, x0, epsilon, max_iter, type_norme="inf", p_norm=None):

    D = np.diag(np.diag(A))
    E = -np.tril(A, -1)
    F = -np.triu(A, 1)

    DE_inv = np.linalg.inv(D - E)

    Gs = DE_inv @ F
    c = DE_inv @ b

    M_norm = choisir_norme(Gs, type_norme, p_norm)
    p = rayon_spectral(Gs)

    historique_X = []
    historique_erreurs = []
    X = x0.copy()
    convergence = False
    nb_iterations = 0


    for i in range(max_iter):
        X_new = Gs @ X + c
        historique_X.append(X_new.copy())

        diff = np.linalg.norm(X_new - X)
        if M_norm < 1:
            err_estime = diff * (M_norm / (1 - M_norm))
            historique_erreurs.append(err_estime)
            critere = err_estime
        else:
            err_estime = diff
            historique_erreurs.append(err_estime)
            critere = diff


        if critere < epsilon:
            convergence = True
            nb_iterations = i + 1
            break

        X = X_new
    if not convergence:
        nb_iterations = max_iter
    solution = historique_X[-1]
   
    return convergence, solution, historique_X, historique_erreurs, p, Gs, M_norm




def analyser_matrice(A):
    res = {}
    res["DDS"] = is_DDS(A)
    res["Symétrique"] = is_symmetric(A)
    res["Définie positive"] = is_positive_definite(A)
    
    return res

def recommander_methodes(A):
    info = analyser_matrice(A)
    recs = []
    if info["DDS"]:
        recs.append("Jacobi recommandé (matrice DDS → convergence garantie)")
        recs.append("Gauss-Seidel recommandé (matrice DDS → convergence garantie)")
    
    if info["Symétrique"] and info["Définie positive"]:
        recs.append("Gauss-Seidel recommandé (matrice symétrique et définie positive → convergence garantie)")
    if not recs:
        recs.append("Convergence non garantie — vérifiez le rayon spectral.")
    return recs 

def afficher_recommandations(A):
    recommandations, info = recommander_methodes(A)
    print("Analyse de la matrice A :")
    for prop, val in info.items():
        print(f" - {prop} : {'Oui' if val else 'Non'}")
    
    if recommandations:
        print("\nMéthodes recommandées pour résoudre Ax=b :")
        for methode in recommandations:
            print(f" - {methode}")
    else:
        print("\nAucune méthode itérative recommandée pour cette matrice.")

def comparer_methodes(A, b, x0, max_iter=100, epsilon=1e-6):
    
    print("\n--- Comparaison Jacobi vs Gauss-Seidel ---")

    # Jacobi
    Js, suite_jacobi, p_j = jacobi(A, b, x0, max_iter, epsilon, visualiser=False)
    X_j = suite_jacobi[-1]
    err_j = np.linalg.norm(A @ X_j - b)

    # Gauss-Seidel
    Gs, suite_gs, p_gs = gauss_seidel(A, b, x0, max_iter, epsilon, visualiser=False)
    X_gs = suite_gs[-1]
    err_gs = np.linalg.norm(A @ X_gs - b)

    print("\n--- Résultats ---")
    print(f"Jacobi : p = {p_j}, erreur = {err_j}")
    print(f"Gauss-Seidel : p = {p_gs}, erreur = {err_gs}")

    # Comparaison vitesse
    if p_j < p_gs:
        print("Jacobi converge plus rapidement (p plus petit)")
    else:
        print("Gauss-Seidel converge plus rapidement")

    # Comparaison précision
    if err_j < err_gs:
        print("Jacobi est plus précis")
    else:
        print("Gauss-Seidel est plus précis")

    # Recommandation finale
    if err_j < err_gs and p_j <= p_gs:
        print("\n Méthode recommandée : Jacobi")
    else:
        print("\n Méthode recommandée : Gauss-Seidel")

def visualiser_convergence(suite_iterative, titre="Convergence", methode="Jacobi"):
    # Calculer l'erreur relative entre itérations successives
    erreurs = []
    for i in range(len(suite_iterative) - 1):
        err = np.linalg.norm(suite_iterative[i+1] - suite_iterative[i])
        erreurs.append(err)
    plt.close('all')
    # Créer le graphique
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(erreurs)+1), erreurs, 'b-o', linewidth=2, markersize=4)
    plt.title(f"{titre} - {methode}", fontsize=14)
    plt.xlabel("Numéro d'itération", fontsize=12)
    plt.ylabel("Erreur entre deux itérations consécutives", fontsize=12)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Afficher la dernière erreur
    if erreurs:
        plt.text(0.7, 0.95, f"Erreur finale: {erreurs[-1]:.2e}", 
                 transform=plt.gca().transAxes, fontsize=10)
    
    plt.show(block=False)

def visualiser_solution(suite_iterative, b, A, titre="Convergence"):
    residus = []
    for x in suite_iterative:
        residu = np.linalg.norm(A @ x - b)
        residus.append(residu)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(len(residus)), residus, 'r-o', linewidth=2, markersize=4)
    plt.title(f"{titre} - Évolution du résidu(erreur)", fontsize=14)
    plt.xlabel("Numéro d'itération", fontsize=12)
    plt.ylabel("||Ax - b||", fontsize=12)
    plt.show(block=False) 


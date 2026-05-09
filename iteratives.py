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

def comparer_methodes(A, b, x0, max_iter=100, epsilon=1e-6, type_norme=2, p_norm=None):
    print("\n--- Comparaison Jacobi vs Gauss-Seidel ---")

    conv_j, X_j, hist_j, errs_j, p_j, Js, _ = jacobi(A, b, x0, epsilon, max_iter, type_norme, p_norm)
    conv_gs, X_gs, hist_gs, errs_gs, p_gs, Gs, _ = gauss_seidel(A, b, x0, epsilon, max_iter, type_norme, p_norm)
    iter_j  = len(hist_j)
    iter_gs = len(hist_gs)
    
    err_j  = float(np.linalg.norm(A @ X_j  - b)) if conv_j  else float('inf')
    err_gs = float(np.linalg.norm(A @ X_gs - b)) if conv_gs else float('inf')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('white')

    # ── Graphe convergence ──
    ax1 = axes[0]
    ax1.set_facecolor('white')
    if errs_j and len(errs_j) > 1:
        ax1.semilogy(range( 1,len(errs_j)+1), [max(e, 1e-16) for e in errs_j],
                     color='#4fc3f7', lw=2, marker='o', markersize=3, label='Jacobi')
    if errs_gs and len(errs_gs) > 1:
        ax1.semilogy(range( 1,len(errs_gs)+1), [max(e, 1e-16) for e in errs_gs],
                     color='#80cbc4', lw=2, marker='^', markersize=3, label='Gauss-Seidel')
    ax1.set_title("Convergence des erreurs", color='black', fontsize=11)
    ax1.set_xlabel("Itération", color='black', fontsize=9)
    ax1.set_ylabel("Erreur estimée", color='black', fontsize=9)
    ax1.tick_params(colors='black', labelsize=8)
    for s in ax1.spines.values():
        s.set_edgecolor('black')
    ax1.legend(facecolor='white', edgecolor='black',
               labelcolor='black', fontsize=9)

    # ── Tableau ──
    ax2 = axes[1]
    ax2.axis('off')
    data = [
        ['Jacobi', 'Oui' if conv_j else 'Non', str(iter_j), f'{err_j:.2e}', f'{p_j:.4f}'],
        ['Gauss-Seidel', 'Oui' if conv_gs else 'Non', str(iter_gs), f'{err_gs:.2e}', f'{p_gs:.4f}'],
    ]
    headers = ['Méthode', 'Convergé', 'Itérations', 'Résidu ‖Ax-b‖', 'ρ spectral']
    table = ax2.table(cellText=data, colLabels=headers,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('white')
            cell.set_text_props(color='black')
        cell.set_edgecolor('black')
    ax2.set_title("Résumé de la comparaison", color='black', fontsize=11)

    #REMARQUE :
    fig.subplots_adjust(bottom=0.25)  # 
    remarques = []
    
    # Comparaison de vitesse
    if iter_gs < iter_j:
        rapport = iter_j / iter_gs
        remarques.append(f"• Gauss-Seidel est {rapport:.1f}x plus rapide que Jacobi ({iter_gs} vs {iter_j} itérations)")
    elif iter_j < iter_gs:
        rapport = iter_gs / iter_j
        remarques.append(f"• Jacobi est {rapport:.1f}x plus rapide que Gauss-Seidel ({iter_j} vs {iter_gs} itérations)")
    else:
        remarques.append(f"• Les deux méthodes convergent en {iter_j} itérations")
    
    # Comparaison des rayons spectraux
    if p_gs < p_j:
        rapport_rho = p_j / p_gs
        remarques.append(f"• Rayon spectral de GS ({p_gs:.4f}) est {rapport_rho:.1f}x plus petit que celui de Jacobi ({p_j:.4f}) → convergence plus rapide")
    elif p_j < p_gs:
        rapport_rho = p_gs / p_j
        remarques.append(f"• Rayon spectral de Jacobi ({p_j:.4f}) est {rapport_rho:.1f}x plus petit que celui de GS ({p_gs:.4f})")
    else:
        remarques.append(f"• Rayons spectraux identiques : {p_j:.4f}")
    
    # Comparaison des résidus
    if err_gs < err_j:
        rapport_err = err_j / err_gs
        remarques.append(f"• GS donne un résidu {rapport_err:.1f}x plus petit que Jacobi ({err_gs:.2e} vs {err_j:.2e})")
    elif err_j < err_gs:
        rapport_err = err_gs / err_j
        remarques.append(f"• Jacobi donne un résidu {rapport_err:.1f}x plus petit que GS ({err_j:.2e} vs {err_gs:.2e})")
    else:
        remarques.append(f"• Résidus identiques : {err_j:.2e}")
    
    # Recommandation finale
    if iter_gs < iter_j and err_gs < err_j:
        remarques.append("Conclusion : Gauss-Seidel est RECOMMANDÉ pour cette matrice")
    elif iter_j < iter_gs and err_j < err_gs:
        remarques.append("Conclusion : Jacobi est RECOMMANDÉ pour cette matrice")
    else:
        remarques.append("⚠️ Les deux méthodes sont comparables, à vous de choisir")
    
    # Affichage des remarques
    text_remarks = "\n".join(remarques)
    ax2.text(0.5, 0.2, text_remarks, transform=ax2.transAxes, fontsize=9, color='black',
             verticalalignment='top', horizontalalignment='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='#f0f0f0', edgecolor='black'))
    
    ax2.set_title("Résumé de la comparaison", color='black', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

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


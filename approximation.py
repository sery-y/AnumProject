"""
==========================================================
AXE 3 : INTERPOLATION ET APPROXIMATION
==========================================================
Module complet contenant :
    1. Normes discrètes et continues (Lp, p=1,2,...,inf)
    2. Interpolation polynomiale (Newton, Lagrange)
    3. Approximation discrète (moindres carrés)
    4. Approximation continue (moindres carrés)
    5. Descente de gradient
    6. Comparaison et sélection du meilleur polynôme
    7. Visualisation (graphes, tableaux d'erreurs)
==========================================================
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad
from math import inf


# ==========================================================
#                 EVALUATION D'UN POLYNOME
# ==========================================================

def evaluer_polynome(coeffs, x):
    """
    Evalue P(x) = a0 + a1*x + a2*x^2 + ...
    Fonctionne avec un scalaire ou un tableau numpy.
    """
    x = np.asarray(x, dtype=float)
    resultat = np.zeros_like(x)
    for i in range(len(coeffs)):
        resultat = resultat + coeffs[i] * (x ** i)
    return resultat


# ==========================================================
#                 NORMES DISCRETES
# ==========================================================

def norme_discrete(vecteur, p=2):
    """
    Calcule la norme Lp d'un vecteur.
    
    Paramètres :
        vecteur : liste ou tableau de nombres
        p       : 1, 2, 3, ..., ou inf
    
    Retourne :
        la norme Lp du vecteur
    """
    vecteur = np.array(vecteur, dtype=float)

    if p == 1:
        return np.sum(np.abs(vecteur))

    elif p == 2:
        return np.sqrt(np.sum(vecteur ** 2))

    elif p == inf:
        return np.max(np.abs(vecteur))

    elif p > 0:
        return (np.sum(np.abs(vecteur) ** p)) ** (1.0 / p)

    else:
        raise ValueError("p doit etre strictement positif.")


# ==========================================================
#                 NORMES CONTINUES
# ==========================================================

def norme_continue(f, a, b, p=2):
    """
    Calcule la norme Lp d'une fonction f sur [a, b].
    
    Paramètres :
        f : fonction (callable)
        a : borne inférieure
        b : borne supérieure
        p : 1, 2, 3, ..., ou inf
    
    Retourne :
        la norme Lp de f sur [a, b]
    """
    if p == 1:
        resultat, _ = quad(lambda x: abs(f(x)), a, b)
        return resultat

    elif p == 2:
        resultat, _ = quad(lambda x: abs(f(x)) ** 2, a, b)
        return np.sqrt(resultat)

    elif p == inf:
        xs = np.linspace(a, b, 10000)
        return max(abs(f(xi)) for xi in xs)

    elif p > 0:
        resultat, _ = quad(lambda x: abs(f(x)) ** p, a, b)
        return resultat ** (1.0 / p)

    else:
        raise ValueError("p doit etre strictement positif.")


# ==========================================================
#              INTERPOLATION DE NEWTON
# ==========================================================

def interpolation_newton(x_points, y_points):
    """
    Calcule le polynôme d'interpolation de Newton.
    
    Paramètres :
        x_points : liste des abscisses
        y_points : liste des ordonnées
    
    Retourne :
        dictionnaire contenant :
            "fonction"     : fonction numérique P(x)
            "expression"   : expression symbolique
            "coefficients" : coefficients des différences divisées
    """
    x_points = np.array(x_points, dtype=float)
    y_points = np.array(y_points, dtype=float)

    x = sp.Symbol('x')
    n = len(x_points)

    # Tableau des différences divisées
    coef = np.zeros((n, n))
    coef[:, 0] = y_points

    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (
                (coef[i + 1][j - 1] - coef[i][j - 1])
                / (x_points[i + j] - x_points[i])
            )

    # Construction du polynôme symbolique
    P_expr = coef[0, 0]

    for i in range(1, n):
        terme = coef[0, i]
        for j in range(i):
            terme *= (x - x_points[j])
        P_expr += terme

    P_expr = sp.expand(P_expr)
    P_func = sp.lambdify(x, P_expr, 'numpy')

    return {
        "fonction": P_func,
        "expression": P_expr,
        "coefficients": coef[0, :]
    }


# ==========================================================
#              INTERPOLATION DE LAGRANGE
# ==========================================================

def interpolation_lagrange(x_points, y_points):
    """
    Calcule le polynôme d'interpolation de Lagrange.
    
    Paramètres :
        x_points : liste des abscisses
        y_points : liste des ordonnées
    
    Retourne :
        dictionnaire contenant :
            "fonction"   : fonction numérique P(x)
            "expression" : expression symbolique
            "base"       : liste des polynômes de base Li
    """
    x_points = np.array(x_points, dtype=float)
    y_points = np.array(y_points, dtype=float)

    x = sp.Symbol('x')
    n = len(x_points)

    P_expr = 0
    base_lagrange = []

    for i in range(n):

        Li = 1

        for j in range(n):
            if i != j:
                Li *= (
                    (x - x_points[j])
                    / (x_points[i] - x_points[j])
                )

        Li = sp.expand(Li)
        base_lagrange.append(Li)
        P_expr += y_points[i] * Li

    P_expr = sp.expand(P_expr)
    P_func = sp.lambdify(x, P_expr, 'numpy')

    return {
        "fonction": P_func,
        "expression": P_expr,
        "base": base_lagrange
    }


# ==========================================================
#         VISUALISATION DES INTERPOLATIONS
# ==========================================================

def tracer_interpolations(x_points, y_points, polynomes):
    """
    Trace le nuage de points et les polynômes d'interpolation.
    
    Paramètres :
        x_points  : liste des abscisses
        y_points  : liste des ordonnées
        polynomes : liste de tuples (fonction, nom)
    """
    X = np.linspace(min(x_points), max(x_points), 500)

    plt.figure(figsize=(10, 6))

    plt.scatter(
        x_points, y_points,
        color='red', s=80, zorder=5,
        label='Points donnés'
    )

    styles = ['-', '--', '-.', ':']
    colors = ['blue', 'green', 'purple', 'orange']

    for i, (P_func, nom) in enumerate(polynomes):

        Y = P_func(X)

        plt.plot(
            X, Y,
            color=colors[i % len(colors)],
            linestyle=styles[i % len(styles)],
            linewidth=2,
            label=f'Interpolation {nom}'
        )

    plt.title("Comparaison des interpolations")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.show()


# ==========================================================
#         TABLEAU D'ERREURS INTERPOLATION
# ==========================================================

def tableau_erreurs_interpolation(x_points, y_points, polynomes):
    """
    Affiche un tableau d'erreurs pour les interpolations.
    
    Paramètres :
        x_points  : liste des abscisses
        y_points  : liste des ordonnées
        polynomes : liste de tuples (fonction, nom)
    """
    print("\n" + "=" * 70)
    print("TABLEAU DES ERREURS - INTERPOLATION")
    print("=" * 70)

    header = f"{'x':>8} {'y':>10}"
    for _, nom in polynomes:
        header += f" {nom+'(x)':>12} {'Erreur':>12}"
    print(header)
    print("-" * 70)

    for xi, yi in zip(x_points, y_points):

        ligne = f"{xi:8.4f} {yi:10.4f}"

        for P_func, _ in polynomes:
            val = float(P_func(xi))
            err = abs(yi - val)
            ligne += f" {val:12.4f} {err:12.2e}"

        print(ligne)

    print("=" * 70)


# ==========================================================
#         MATRICE M - CAS DISCRET
# ==========================================================

def construire_matrice_discrete(x, degre):
    """
    Construit la matrice M du système M*A = b (cas discret).
    M[j][k] = Σ xi^(j+k)
    """
    x = np.array(x, dtype=float)
    s = degre
    M = np.zeros((s + 1, s + 1))

    for j in range(s + 1):
        for k in range(s + 1):
            M[j][k] = np.sum(x ** (j + k))

    return M


# ==========================================================
#         VECTEUR b - CAS DISCRET
# ==========================================================

def construire_vecteur_discret(x, y, degre):
    """
    Construit le vecteur b du système M*A = b (cas discret).
    b[k] = Σ xi^k * yi
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    s = degre
    b = np.zeros(s + 1)

    for k in range(s + 1):
        b[k] = np.sum((x ** k) * y)

    return b


# ==========================================================
#         APPROXIMATION DISCRETE
# ==========================================================

def approximation_discrete(x, y, degre):
    """
    Approximation polynomiale au sens des moindres carrés (cas discret).
    
    Paramètres :
        x     : liste des abscisses
        y     : liste des ordonnées
        degre : degré maximal du polynôme (s)
    
    Retourne :
        coefficients [a0, a1, ..., as] du polynôme
    """
    M = construire_matrice_discrete(x, degre)
    b = construire_vecteur_discret(x, y, degre)
    A = np.linalg.solve(M, b)
    return A


# ==========================================================
#         MATRICE M - CAS CONTINU
# ==========================================================

def construire_matrice_continue(a, b, degre):
    """
    Construit la matrice M du système M*A = b (cas continu).
    M[j][k] = ∫ x^(j+k) dx sur [a, b]
    """
    s = degre
    M = np.zeros((s + 1, s + 1))

    for j in range(s + 1):
        for k in range(s + 1):
            integrale, _ = quad(lambda x: x ** (j + k), a, b)
            M[j][k] = integrale

    return M


# ==========================================================
#         VECTEUR b - CAS CONTINU
# ==========================================================

def construire_vecteur_continue(f, a, b, degre):
    """
    Construit le vecteur b du système M*A = b (cas continu).
    b[k] = ∫ x^k * f(x) dx sur [a, b]
    """
    s = degre
    B = np.zeros(s + 1)

    for k in range(s + 1):
        integrale, _ = quad(lambda x: (x ** k) * f(x), a, b)
        B[k] = integrale

    return B


# ==========================================================
#         APPROXIMATION CONTINUE
# ==========================================================

def approximation_continue(f, a, b, degre):
    """
    Approximation polynomiale au sens des moindres carrés (cas continu).
    
    Paramètres :
        f     : fonction à approximer (callable)
        a     : borne inférieure
        b     : borne supérieure
        degre : degré maximal du polynôme (s)
    
    Retourne :
        coefficients [a0, a1, ..., as] du polynôme
    """
    M = construire_matrice_continue(a, b, degre)
    B = construire_vecteur_continue(f, a, b, degre)
    A = np.linalg.solve(M, B)
    return A


# ==========================================================
#         ERREUR DISCRETE
# ==========================================================

def erreur_discrete(x, y, coeffs):
    """
    Calcule l'erreur en chaque point : yi - P(xi).
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    erreurs = []
    for i in range(len(x)):
        val = evaluer_polynome(coeffs, x[i])
        erreurs.append(y[i] - float(val))

    return erreurs


# ==========================================================
#         ERREUR CONTINUE
# ==========================================================

def erreur_continue(f, coeffs, a, b):
    """
    Calcule l'erreur L2 continue : ∫ (f(x) - P(x))^2 dx
    """
    erreur, _ = quad(
        lambda x: (f(x) - float(evaluer_polynome(coeffs, x))) ** 2,
        a, b
    )
    return erreur


# ==========================================================
#         DESCENTE DE GRADIENT
# ==========================================================

def descente_gradient(x, y, taux=0.001, iterations=1000):
    """
    Régression linéaire par descente de gradient.
    y = a0 + a1*x
    
    Retourne :
        coefficients [a0, a1]
        historique des coefficients
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    a0 = 0.0
    a1 = 0.0
    n = len(x)
    historique = []

    for _ in range(iterations):

        grad_a0 = 0.0
        grad_a1 = 0.0

        for i in range(n):
            prediction = a0 + a1 * x[i]
            erreur = prediction - y[i]
            grad_a0 += erreur
            grad_a1 += erreur * x[i]

        grad_a0 = (2.0 / n) * grad_a0
        grad_a1 = (2.0 / n) * grad_a1

        a0 = a0 - taux * grad_a0
        a1 = a1 - taux * grad_a1

        historique.append((a0, a1))

    return [a0, a1], historique


# ==========================================================
#    COMPARAISON DE PLUSIEURS APPROXIMATIONS DISCRETES
# ==========================================================

def comparer_approximations_discretes(x, y, degres, p=2):
    """
    Compare plusieurs polynômes d'approximation de degrés différents.
    
    Paramètres :
        x      : liste des abscisses
        y      : liste des ordonnées
        degres : liste des degrés à tester [1, 2, 3, ...]
        p      : norme Lp utilisée (1, 2, ..., inf)
    
    Retourne :
        resultats      : dictionnaire {degré: {coeffs, erreur_globale, erreurs_points}}
        meilleur_degre : le degré optimal
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    resultats = {}
    meilleur_degre = None
    meilleure_erreur = float('inf')

    for degre in degres:

        # Calcul des coefficients
        coeffs = approximation_discrete(x, y, degre)

        # Erreurs point par point
        erreurs_pts = []
        for i in range(len(x)):
            val = float(evaluer_polynome(coeffs, x[i]))
            erreurs_pts.append(abs(y[i] - val))

        # Erreur globale selon la norme p
        erreur_globale = norme_discrete(erreurs_pts, p)

        resultats[degre] = {
            "coeffs": coeffs,
            "erreur_globale": erreur_globale,
            "erreurs_points": erreurs_pts
        }

        # Détection du meilleur
        if erreur_globale < meilleure_erreur:
            meilleure_erreur = erreur_globale
            meilleur_degre = degre

    return resultats, meilleur_degre


# ==========================================================
#   VISUALISATION COMPARAISON APPROXIMATIONS
# ==========================================================

def tracer_comparaison_approximations(
        x_points, y_points, resultats, meilleur_degre, p
):
    """
    Trace le nuage de points + tous les polynômes d'approximation.
    Le meilleur polynôme est tracé en ROUGE EPAIS.
    Les erreurs sont affichées comme segments verticaux.
    
    Paramètres :
        x_points       : liste des abscisses
        y_points       : liste des ordonnées
        resultats      : dictionnaire des résultats
        meilleur_degre : degré optimal
        p              : norme utilisée
    """
    x_points = np.array(x_points, dtype=float)
    y_points = np.array(y_points, dtype=float)

    X = np.linspace(min(x_points), max(x_points), 500)

    plt.figure(figsize=(12, 7))

    # Nuage de points
    plt.scatter(
        x_points, y_points,
        color='black', s=100, zorder=10,
        label='Nuage de points', edgecolors='white'
    )

    colors = ['blue', 'green', 'purple', 'orange', 'brown', 'cyan']
    styles = ['--', '-.', ':', '--']

    for i, (degre, data) in enumerate(resultats.items()):

        coeffs = data["coeffs"]
        erreur = data["erreur_globale"]
        Y = evaluer_polynome(coeffs, X)

        if degre == meilleur_degre:

            # Meilleur polynôme en ROUGE EPAIS
            plt.plot(
                X, Y,
                color='red',
                linewidth=3.5,
                zorder=8,
                label=(
                    f'Degré {degre} - MEILLEUR '
                    f'(Erreur L{p} = {erreur:.4f})'
                )
            )

            # Segments d'erreur verticaux
            for j in range(len(x_points)):

                val = float(evaluer_polynome(coeffs, x_points[j]))

                plt.plot(
                    [x_points[j], x_points[j]],
                    [y_points[j], val],
                    color='red',
                    linewidth=1.5,
                    linestyle='-',
                    alpha=0.6
                )

                # Annotation de l'erreur
                err_j = abs(y_points[j] - val)
                mid_y = (y_points[j] + val) / 2

                plt.annotate(
                    f'{err_j:.2f}',
                    xy=(x_points[j], mid_y),
                    fontsize=7,
                    color='red',
                    ha='left'
                )

        else:

            # Autres polynômes
            color = colors[i % len(colors)]
            style = styles[i % len(styles)]

            plt.plot(
                X, Y,
                color=color,
                linestyle=style,
                linewidth=1.8,
                alpha=0.7,
                label=(
                    f'Degré {degre} '
                    f'(Erreur L{p} = {erreur:.4f})'
                )
            )

    plt.title(
        f"Approximation polynomiale - "
        f"Meilleur polynôme : degré {meilleur_degre} "
        f"(norme L{p})",
        fontsize=13
    )
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.show()


# ==========================================================
#     TABLEAU D'ERREURS APPROXIMATION
# ==========================================================

def tableau_erreurs_approximation(
        x_points, y_points, resultats, meilleur_degre
):
    """
    Affiche un tableau d'erreurs pour chaque polynôme d'approximation.
    Le meilleur est marqué avec ★.
    
    Paramètres :
        x_points       : liste des abscisses
        y_points       : liste des ordonnées
        resultats      : dictionnaire des résultats
        meilleur_degre : degré optimal
    """
    x_points = np.array(x_points, dtype=float)
    y_points = np.array(y_points, dtype=float)

    degres = list(resultats.keys())

    # ========== AFFICHAGE CONSOLE ==========
    print("\n" + "=" * 90)
    print("TABLEAU DES ERREURS - APPROXIMATION")
    print("=" * 90)

    # En-tête
    header = f"{'x':>8} {'y donné':>10}"
    for degre in degres:
        marqueur = " ★" if degre == meilleur_degre else ""
        header += f" {'P'+str(degre)+'(x)'+marqueur:>14} {'|Erreur|':>10}"
    print(header)
    print("-" * 90)

    # Lignes
    for i in range(len(x_points)):

        ligne = f"{x_points[i]:8.3f} {y_points[i]:10.4f}"

        for degre in degres:
            coeffs = resultats[degre]["coeffs"]
            val = float(evaluer_polynome(coeffs, x_points[i]))
            err = abs(y_points[i] - val)
            ligne += f" {val:14.4f} {err:10.2e}"

        print(ligne)

    # Résumé
    print("-" * 90)

    ligne_err = f"{'Erreur globale':>19}"
    for degre in degres:
        err = resultats[degre]["erreur_globale"]
        marqueur = " ★" if degre == meilleur_degre else ""
        ligne_err += f" {err:14.6f} {'':>10}{marqueur}"

    print(ligne_err)
    print("=" * 90)

    print(
        f"\n→ Le meilleur polynôme est de degré {meilleur_degre} "
        f"(erreur = {resultats[meilleur_degre]['erreur_globale']:.6f})"
    )

    # ========== FIGURE MATPLOTLIB ==========
    donnees = []
    headers_tab = ["x", "y donné"]

    for degre in degres:
        marqueur = " ★" if degre == meilleur_degre else ""
        headers_tab.append(f"P{degre}(x){marqueur}")
        headers_tab.append(f"|Erreur|")

    for i in range(len(x_points)):

        ligne = [f"{x_points[i]:.3f}", f"{y_points[i]:.4f}"]

        for degre in degres:
            coeffs = resultats[degre]["coeffs"]
            val = float(evaluer_polynome(coeffs, x_points[i]))
            err = abs(y_points[i] - val)
            ligne.append(f"{val:.4f}")
            ligne.append(f"{err:.2e}")

        donnees.append(ligne)

    fig, ax = plt.subplots(
        figsize=(max(12, 4 * len(degres)), max(3, 0.4 * len(x_points)))
    )
    ax.axis('off')

    table = ax.table(
        cellText=donnees,
        colLabels=headers_tab,
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.4)

    # Colorer les colonnes du meilleur polynôme
    for key, cell in table.get_celld().items():

        row, col = key

        if row == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')

    plt.title(
        f"Tableau des erreurs - Meilleur : degré {meilleur_degre}",
        fontsize=12,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.show()

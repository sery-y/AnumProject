import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# -----------------------------
# Newton
# -----------------------------
def newton_polynomial(x_points, y_points):
    x = sp.Symbol('x')
    n = len(x_points)
    coef = np.zeros([n, n])
    coef[:,0] = y_points
    for j in range(1, n):
        for i in range(n-j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x_points[i+j] - x_points[i])
    P_expr = coef[0,0]
    for i in range(1, n):
        term = coef[0,i]
        for j in range(i):
            term *= (x - x_points[j])
        P_expr += term
    P_expr = sp.expand(P_expr)
    P_func = sp.lambdify(x, P_expr, 'numpy')
    return P_func, P_expr, coef[0,:]

# -----------------------------
# Lagrange
# -----------------------------
def lagrange_polynomial(x_points, y_points):
    x = sp.Symbol('x')
    n = len(x_points)
    L_list = []
    P_expr = 0
    for i in range(n):
        Li = 1
        for j in range(n):
            if j != i:
                Li *= (x - x_points[j])/(x_points[i] - x_points[j])
        Li = sp.simplify(Li)
        L_list.append(Li)
        P_expr += y_points[i] * Li
    P_expr = sp.expand(P_expr)
    P_func = sp.lambdify(x, P_expr, 'numpy')
    return P_func, P_expr, L_list

# -----------------------------
# Graphe comparatif
# -----------------------------

def plot_corbe(x_points, y_points, polynomials):
    X = np.linspace(min(x_points), max(x_points), 400)
    plt.figure(figsize=(10, 6))
    plt.scatter(x_points, y_points, color='red', zorder=5, label='Points donnés')
    

    styles = [
        {'linestyle': '-',  'linewidth': 2.5, 'alpha': 0.7}, 
        {'linestyle': '--', 'linewidth': 1.5, 'alpha': 1.0}  
    ]

    for i, (P_func, name) in enumerate(polynomials):
        Y_interp = np.array(P_func(X), dtype=float)
        
        style = styles[i % len(styles)]
        plt.plot(X, Y_interp, label=f'Interpolation {name}', **style)
    
    plt.legend()
    plt.title("Comparaison des interpolations")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()




# -----------------------------
# Tableau d'erreurs 
# -----------------------------
def error_table_points(x_points, y_points, polynomials):
    table_data = []
    headers = ["x", "y (donné)"]
    for _, name in polynomials:
        headers += [f"{name}(x)", f"Erreur {name}"]
    
    for xi, yi in zip(x_points, y_points):
        row = [f"{xi:.2f}", f"{yi:.4f}"]
        for P_func, _ in polynomials:
            val = P_func(xi)
            err = abs(yi - val)
            row += [f"{val:.4f}", f"{err:.4e}"]
        table_data.append(row)
    
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')
    table = ax.table(cellText=table_data, colLabels=headers, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title("Tableau d'erreurs ")
    plt.show()
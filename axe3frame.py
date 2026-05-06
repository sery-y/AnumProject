"""
axe3_frame.py — Axe 3 : Interpolation & Approximation.
Méthodes : Lagrange, Newton différences divisées, Moindres carrés, Descente gradient.
Dépendances : shared.py
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv

from interfacePartagee import C, F, Card, StatCard, mk_entry, mk_btn


# ══════════════════════════════════════════════
#  ALGORITHMES D'INTERPOLATION / APPROXIMATION
# ══════════════════════════════════════════════

def lagrange_eval(xp, yp, xi):
    """Évalue le polynôme de Lagrange en xi."""
    n = len(xp)
    result = 0.0
    for i in range(n):
        Li = 1.0
        for j in range(n):
            if j != i:
                Li *= (xi - xp[j]) / (xp[i] - xp[j])
        result += yp[i] * Li
    return result


def newton_interp_eval(xp, yp, xi):
    """Évalue le polynôme de Newton (différences divisées) en xi."""
    n = len(xp)
    # Tableau des différences divisées
    dd = yp.copy().astype(float)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            dd[i] = (dd[i] - dd[i - 1]) / (xp[i] - xp[i - j])
    # Évaluation par le schéma de Horner
    result = dd[n - 1]
    for i in range(n - 2, -1, -1):
        result = result * (xi - xp[i]) + dd[i]
    return result


def mc_discret(xp, yp, deg):
    """Moindres carrés discrets — retourne un np.poly1d de degré deg."""
    coeffs = np.polyfit(xp, yp, deg)
    return np.poly1d(coeffs)


def descente_gradient_lin(xp, yp, lr=0.01, niter=10_000):
    """Régression linéaire y = m·x + b par descente de gradient.
    Retourne (m, b)."""
    m, b = 0.0, 0.0
    n = len(xp)
    for _ in range(niter):
        y_pred = m * xp + b
        err    = y_pred - yp
        m -= lr * (2 / n) * np.dot(err, xp)
        b -= lr * (2 / n) * err.sum()
    return m, b


# ══════════════════════════════════════════════
#  FRAME PRINCIPALE
# ══════════════════════════════════════════════

class Axe3Frame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=C["bg"])
        self._exp_data = {}
        self._build()

    # ──────────────────────────────────────────
    def _build(self):
        # ── Top bar ──
        top = tk.Frame(self, bg=C["bg"], padx=28, pady=18)
        top.pack(fill="x")
        tk.Label(top, text="Axe 3 — Interpolation & Approximation",
                 font=F["title"], bg=C["bg"], fg=C["white"]).pack(side="left")
        tk.Label(top, text=" Étudiant 3 ", font=F["small"],
                 bg="#2a1f0a", fg=C["amber"], padx=8, pady=4).pack(side="right", anchor="n")
        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

        # ── Scrollable canvas ──
        outer = tk.Canvas(self, bg=C["bg"], bd=0, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=outer.yview)
        outer.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        outer.pack(fill="both", expand=True)
        content = tk.Frame(outer, bg=C["bg"])
        win = outer.create_window((0, 0), window=content, anchor="nw")
        outer.bind("<Configure>", lambda e: outer.itemconfig(win, width=e.width))
        content.bind("<Configure>",
                     lambda e: outer.configure(scrollregion=outer.bbox("all")))

        p = dict(padx=28, pady=8)

        # ── Méthode ──
        mc_card = Card(content, "Méthode")
        mc_card.pack(fill="x", **p)
        grid = tk.Frame(mc_card.body, bg=C["card"])
        grid.pack(fill="x")
        grid.columnconfigure((0, 1), weight=1)

        self.method_var = tk.StringVar(value="Lagrange")
        self._meth_frames = {}
        methods = [
            ("Lagrange",          "Interpolation polynomiale", 0, 0),
            ("Newton diff. div.", "Interpolation Newton",      0, 1),
            ("Moindres carrés",   "Approximation discrète",    1, 0),
            ("Descente gradient", "Régression linéaire",       1, 1),
        ]
        for name, desc, r, c in methods:
            f = tk.Frame(grid, bg=C["card2"],
                         highlightthickness=1, highlightbackground=C["border"],
                         padx=14, pady=12, cursor="hand2")
            f.grid(row=r, column=c, padx=5, pady=5, sticky="ew")
            nl = tk.Label(f, text=name, font=F["h3"], bg=C["card2"], fg=C["white"])
            nl.pack(anchor="w")
            dl = tk.Label(f, text=desc, font=F["small"], bg=C["card2"], fg=C["gray"])
            dl.pack(anchor="w")
            for w in [f, nl, dl]:
                w.bind("<Button-1>", lambda e, n=name: self._sel_meth(n))
            self._meth_frames[name] = f

        # ── Recommandation ──
        rec = tk.Frame(content, bg=C["teal_bg"],
                       highlightthickness=1, highlightbackground=C["teal"],
                       padx=16, pady=12)
        rec.pack(fill="x", **p)
        tk.Label(rec, text="Recommandation", font=F["h3"],
                 bg=C["teal_bg"], fg=C["teal"]).pack(anchor="w")
        self._rec3 = tk.Label(rec, text="", font=F["body"],
                              bg=C["teal_bg"], fg="#b0e8d5",
                              wraplength=900, justify="left")
        self._rec3.pack(anchor="w", pady=(4, 0))
        self._sel_meth("Lagrange")

        # ── Données ──
        dc = Card(content, "Données")
        dc.pack(fill="x", **p)
        row = tk.Frame(dc.body, bg=C["card"])
        row.pack(fill="x")
        for lbl, attr, default, w in [
            ("x (virgule)", "e_xp",  "0,1,2,3,4",  18),
            ("y (virgule)", "e_yp",  "1,4,9,16,25", 18),
            ("x à évaluer", "e_xi",  "2.5",          7),
            ("Degré",       "e_deg", "2",            4),
        ]:
            col = tk.Frame(row, bg=C["card"])
            col.pack(side="left", padx=(0, 10))
            tk.Label(col, text=lbl, font=F["small"],
                     bg=C["card"], fg=C["gray"]).pack(anchor="w", pady=(0, 3))
            e = mk_entry(col, default, w)
            e.pack(ipady=5)
            setattr(self, attr, e)

        col_b = tk.Frame(row, bg=C["card"])
        col_b.pack(side="left")
        tk.Label(col_b, text=" ", font=F["small"], bg=C["card"], fg=C["gray"]).pack(pady=(0, 3))
        mk_btn(col_b, "Calculer ↗", self._run3, C["amber"]).pack(ipady=4)

        # ── Stats ──
        sr = tk.Frame(content, bg=C["bg"])
        sr.pack(fill="x", **p)
        sr.columnconfigure((0, 1, 2), weight=1)
        self.st3_v = StatCard(sr, "Valeur interpolée")
        self.st3_v.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.st3_m = StatCard(sr, "Méthode")
        self.st3_m.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.st3_p = StatCard(sr, "Nb points")
        self.st3_p.grid(row=0, column=2, sticky="ew")

        # ── Graphe ──
        gc = Card(content, "Visualisation")
        gc.pack(fill="x", **p)
        self.fig3, self.ax3 = plt.subplots(figsize=(9, 3.2), facecolor=C["card"])
        self.ax3.set_facecolor(C["bg"])
        for sp_ in self.ax3.spines.values():
            sp_.set_edgecolor(C["border"])
        self.ax3.tick_params(colors=C["gray"], labelsize=8)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, gc.body)
        self.canvas3.get_tk_widget().pack(fill="both")

        # ── Export / Import ──
        ec = Card(content, "Export / Import")
        ec.pack(fill="x", **p)
        btns = tk.Frame(ec.body, bg=C["card"])
        btns.pack(fill="x")
        mk_btn(btns, "⬇ Exporter CSV",         self._export3,  secondary=True).pack(side="left", padx=(0, 8))
        mk_btn(btns, "⬆ Importer points CSV",   self._import3,  secondary=True).pack(side="left")

        tk.Frame(content, bg=C["bg"], height=20).pack()

    # ──────────────────────────────────────────
    def _sel_meth(self, name):
        self.method_var.set(name)
        for n, f in self._meth_frames.items():
            is_sel = (n == name)
            bg  = C["acc_bg"] if is_sel else C["card2"]
            brd = C["accent"] if is_sel else C["border"]
            f.configure(bg=bg, highlightbackground=brd)
            for w in f.winfo_children():
                w.configure(bg=bg)
        recs = {
            "Lagrange":          "Lagrange est simple mais peut osciller (phénomène de Runge) avec beaucoup de points.",
            "Newton diff. div.": "Newton par différences divisées est plus stable numériquement.",
            "Moindres carrés":   "Moindres carrés minimise la somme des carrés des résidus. Idéal pour données bruitées.",
            "Descente gradient": "Descente de gradient optimise itérativement la droite de régression.",
        }
        self._rec3.configure(text=recs.get(name, ""))

    # ──────────────────────────────────────────
    def _run3(self):
        try:
            xp  = np.array([float(v) for v in self.e_xp.get().split(",")])
            yp  = np.array([float(v) for v in self.e_yp.get().split(",")])
            xi  = float(self.e_xi.get())
            deg = int(self.e_deg.get())
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            return

        method = self.method_var.get()
        self.ax3.cla()
        self.ax3.set_facecolor(C["bg"])
        for sp_ in self.ax3.spines.values():
            sp_.set_edgecolor(C["border"])
        self.ax3.tick_params(colors=C["gray"], labelsize=8)

        x_d = np.linspace(min(xp) - 0.5, max(xp) + 0.5, 400)
        val = None

        if method == "Lagrange":
            y_d = [lagrange_eval(xp, yp, xi_) for xi_ in x_d]
            val = lagrange_eval(xp, yp, xi)
            self.ax3.plot(x_d, y_d, color=C["acc_light"], lw=2, label="Lagrange")

        elif method == "Newton diff. div.":
            y_d = [newton_interp_eval(xp, yp, xi_) for xi_ in x_d]
            val = newton_interp_eval(xp, yp, xi)
            self.ax3.plot(x_d, y_d, color=C["acc_light"], lw=2, label="Newton")

        elif method == "Moindres carrés":
            poly = mc_discret(xp, yp, deg)
            y_d  = poly(x_d)
            val  = poly(xi)
            self.ax3.plot(x_d, y_d, color=C["acc_light"], lw=2, label=f"MC deg {deg}")

        elif method == "Descente gradient":
            m, b = descente_gradient_lin(xp, yp)
            y_d  = m * x_d + b
            val  = m * xi + b
            self.ax3.plot(x_d, y_d, color=C["acc_light"], lw=2,
                          label=f"y={m:.3f}x+{b:.3f}")

        self.ax3.scatter(xp, yp, color=C["amber"], zorder=5, s=55, label="Points")
        if val is not None:
            self.ax3.scatter([xi], [val], color=C["teal"], zorder=6, s=80,
                             label=f"f({xi})={val:.4f}")
        self.ax3.set_xlabel("x", color=C["gray"], fontsize=9)
        self.ax3.set_ylabel("y", color=C["gray"], fontsize=9)
        self.ax3.legend(facecolor=C["card"], edgecolor=C["border"],
                        labelcolor=C["white"], fontsize=8)
        self.fig3.tight_layout(pad=0.8)
        self.canvas3.draw()

        self.st3_v.set(f"{val:.6f}" if val is not None else "—")
        self.st3_m.set(method.split()[0])
        self.st3_p.set(str(len(xp)))
        self._exp_data = {
            "xp": list(xp), "yp": list(yp),
            "xi": xi, "val": val, "method": method,
        }

    # ──────────────────────────────────────────
    def _export3(self):
        if not self._exp_data:
            messagebox.showinfo("Info", "Aucun résultat.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialfile="resultats_axe3.csv")
        if not path:
            return
        d = self._exp_data
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["Méthode", d["method"]])
            w.writerow(["x évalué", d["xi"]])
            w.writerow(["Valeur",   d["val"]])
            w.writerow([])
            w.writerow(["x_i", "y_i"])
            for xi, yi in zip(d["xp"], d["yp"]):
                w.writerow([xi, yi])
        messagebox.showinfo("Exporté", f"Sauvegardé :\n{path}")

    def _import3(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV", "*.csv"), ("Tous", "*.*")])
        if not path:
            return
        try:
            xp, yp = [], []
            with open(path, newline='', encoding='utf-8') as f:
                rows = list(csv.reader(f))
            reading = False
            for row in rows:
                if row and row[0] == "x_i":
                    reading = True
                    continue
                if reading and len(row) >= 2:
                    try:
                        xp.append(float(row[0]))
                        yp.append(float(row[1]))
                    except Exception:
                        pass
            if not xp:
                messagebox.showerror("Erreur", "Aucune donnée trouvée.")
                return
            self.e_xp.delete(0, "end")
            self.e_xp.insert(0, ",".join(str(v) for v in xp))
            self.e_yp.delete(0, "end")
            self.e_yp.insert(0, ",".join(str(v) for v in yp))
            messagebox.showinfo("Importé", f"{len(xp)} points chargés.")
        except Exception as e:
            messagebox.showerror("Erreur import", str(e))
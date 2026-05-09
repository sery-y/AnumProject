"""
axe3_frame.py — Axe 3 : Interpolation & Approximation.
Méthodes : Lagrange, Newton différences divisées, Moindres carrés disc/cont, Descente gradient.
Dépendances : interfacePartagee.py, approximation.py
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv
from math import inf

from interfacePartagee import C, F, Card, StatCard, mk_entry, mk_btn

# ══════════════════════════════════════════════
#  IMPORT DES FONCTIONS DU MODULE AXE3
# ══════════════════════════════════════════════
from axe3 import (
    evaluer_polynome,
    norme_discrete,

    interpolation_newton,
    interpolation_lagrange,
    tracer_interpolations,
    tableau_erreurs_interpolation,
    
    approximation_discrete,
    
    approximation_continue,
    erreur_discrete,
    erreur_continue,
    descente_gradient,
    comparer_approximations_discretes,
    tracer_comparaison_approximations,
    tableau_erreurs_approximation,
    recommander_methode,
)


# ══════════════════════════════════════════════
#  FRAME PRINCIPALE
# ══════════════════════════════════════════════

class Axe3Frame(tk.Frame):
    DEGRE_MIN = 1 #pour la comparaison de l approximation
    DEGRE_MAX = 4
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
            ("Lagrange",               "Interpolation polynomiale",   0, 0),
            ("Newton diff. div.",       "Interpolation Newton",        0, 1),
            ("Moindres carrés disc",    "Approximation discrète",      1, 0),
            ("Moindres carrés cont",    "Approximation continue",      1, 1),
            ("Descente gradient",       "Régression linéaire",         2, 0),
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
        self._rec3 = tk.Label(rec, text="Entrez le nuage de points (x et y) pour obtenir une recommandation automatique.", font=F["body"],
                              bg=C["teal_bg"], fg="#b0e8d5",
                              wraplength=900, justify="left")
        self._rec3.pack(anchor="w", pady=(4, 0))
        

        # ── Données ──
        dc = Card(content, "Données")
        dc.pack(fill="x", **p)
        row = tk.Frame(dc.body, bg=C["card"])
        row.pack(fill="x")
        for lbl, attr, default, w in [
            ("x (virgule)",   "e_xp",  "0,1,2,3,4",   18),
            ("y (virgule)",   "e_yp",  "1,4,9,16,25",  18),
            ("x à évaluer",   "e_xi",  "2.5",            7),
            ("Degré",         "e_deg", "2",               4),
            ("Norme p",       "e_p",   "2",               4),
            
        ]:
            col = tk.Frame(row, bg=C["card"])
            col.pack(side="left", padx=(0, 10))
            tk.Label(col, text=lbl, font=F["small"],
                     bg=C["card"], fg=C["gray"]).pack(anchor="w", pady=(0, 3))
            e = mk_entry(col, default, w)
            e.pack(ipady=5)
            setattr(self, attr, e)
        
        self.e_xp.bind("<FocusOut>", self._update_recommendation) #se decleche si lutilisateur clique ailleurs
        self.e_yp.bind("<FocusOut>", self._update_recommendation)
        self.e_xp.bind("<Return>", self._update_recommendation) # se decleche quand il tape entree
        self.e_yp.bind("<Return>", self._update_recommendation)

        col_b = tk.Frame(row, bg=C["card"])
        col_b.pack(side="left")
        tk.Label(col_b, text=" ", font=F["small"], bg=C["card"], fg=C["gray"]).pack(pady=(0, 3))
        mk_btn(col_b, "Calculer ", self._run3, C["amber"]).pack(ipady=4)

        # ── Stats ──
        sr = tk.Frame(content, bg=C["bg"])
        sr.pack(fill="x", **p)
        sr.columnconfigure((0, 1, 2, 3), weight=1)
        self.st3_v   = StatCard(sr, "Valeur calculée")
        self.st3_v.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.st3_m   = StatCard(sr, "Méthode")
        self.st3_m.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.st3_p   = StatCard(sr, "Nb points")
        self.st3_p.grid(row=0, column=2, sticky="ew", padx=(0, 8))
        self.st3_err = StatCard(sr, "Erreur")
        self.st3_err.grid(row=0, column=3, sticky="ew")

        # ── Expression symbolique ──
        expr_card = Card(content, "Expression du polynôme")
        expr_card.pack(fill="x", **p)
        self._lbl_expr = tk.Label(
            expr_card.body, text="—", font=F["body"],
            bg=C["card"], fg=C["acc_light"],
            wraplength=1000, justify="left"
        )
        self._lbl_expr.pack(anchor="w", pady=4)

        # ── Graphe ──
        gc = Card(content, "Visualisation")
        gc.pack(fill="x", **p)
        self.fig3, self.ax3 = plt.subplots(figsize=(10, 4), facecolor=C["card"])
        self.fig3.set_tight_layout(True)
        self.ax3.set_facecolor(C["bg"])
        for sp_ in self.ax3.spines.values():
            sp_.set_edgecolor(C["border"])
        self.ax3.tick_params(colors=C["gray"], labelsize=8)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, gc.body)
        self.canvas3.get_tk_widget().pack(fill="both")

        # ── Boutons supplémentaires ──
        extra_card = Card(content, "Outils avancés")
        extra_card.pack(fill="x", **p)
        btn_row = tk.Frame(extra_card.body, bg=C["card"])
        btn_row.pack(fill="x")
        mk_btn(btn_row, "📊 Tableau d'erreurs",
               self._show_error_table,  secondary=True).pack(side="left", padx=(0, 8))
        mk_btn(btn_row, "📈 Comparer (algo-interpolation) ou (degrés-approximation)",
               self._compare_degrees,   secondary=True).pack(side="left", padx=(0, 8))
        mk_btn(btn_row, "🔢 Normes (L1 / L2 / L∞)",
               self._show_norms,        secondary=True).pack(side="left")

        # ── Export / Import ──
        ec = Card(content, "Export / Import")
        ec.pack(fill="x", **p)
        btns = tk.Frame(ec.body, bg=C["card"])
        btns.pack(fill="x")
        mk_btn(btns, " Exporter CSV",        self._export3, secondary=True).pack(side="left", padx=(0, 8))
        
        self._sel_meth("Lagrange")

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

    # ── Désactiver le champ Degré pour les interpolations ──
      is_interpolation = name in ("Lagrange", "Newton diff. div.")
      state = "disabled" if is_interpolation else "normal"
      self.e_deg.configure(state=state)

    def _update_recommendation(self, event=None):
        
        try:
            x_points = [float(v) for v in self.e_xp.get().split(",")]
            y_points = [float(v) for v in self.e_yp.get().split(",")]
        except Exception:
            self._rec3.configure(text="En attente de données valides...")
            return

        # Appel de la fonction  de axe3.py
        reco_methode, reco_texte = recommander_methode(x_points, y_points)
        
        # Mise à jour du texte de recommandation dans l'interface
        self._rec3.configure(text=reco_texte)

        

    # ──────────────────────────────────────────
    def _parse_inputs(self):
        """Parse et retourne (xp, yp, xi, deg, p_norm, a, b)."""
        xp     = np.array([float(v) for v in self.e_xp.get().split(",")])
        yp     = np.array([float(v) for v in self.e_yp.get().split(",")])
        xi     = float(self.e_xi.get())
        deg    = int(self.e_deg.get())
        p_raw  = self.e_p.get().strip()
        p_norm = inf if p_raw.lower() in ("inf", "∞") else int(p_raw)
        a = float(np.min(xp)) #definir l intervalle pour l approxiamation
        b = float(np.max(xp))
        return xp, yp, xi, deg, p_norm, a, b

    # ──────────────────────────────────────────
    def _run3(self):
        try:
            xp, yp, xi, deg, p_norm, a, b = self._parse_inputs()
        except Exception as e:
            messagebox.showerror("Erreur de saisie", str(e))
            return

        method = self.method_var.get()

        # ── Réinitialisation du graphe ──
        self.ax3.cla()
        self.ax3.set_facecolor(C["bg"])
        for sp_ in self.ax3.spines.values():
            sp_.set_edgecolor(C["border"])
        self.ax3.tick_params(colors=C["gray"], labelsize=8)

        x_d      = np.linspace(min(xp) - 0.5, max(xp) + 0.5, 400)
        val      = None
        expr_txt = "—"
        err_l2   = None
        coeffs   = None

        # ══════════════════════════════════════
        #  APPELS AUX FONCTIONS DU MODULE AXE3
        # ══════════════════════════════════════

        if method == "Lagrange":
            res    = interpolation_lagrange(xp, yp)
            P_func = res["fonction"]
            val    = float(P_func(xi))
            expr_txt = str(res["expression"])
            y_d    = np.array([float(P_func(t)) for t in x_d])
            errs   = [abs(yp[i] - float(P_func(xp[i]))) for i in range(len(xp))]
            err_l2 = norme_discrete(errs, p=p_norm)
            self.ax3.plot(x_d, y_d, color=C["acc_light"], lw=2, label="Lagrange")

        elif method == "Newton diff. div.":
            res    = interpolation_newton(xp, yp)
            P_func = res["fonction"]
            coeffs = res["coefficients"]
            val    = float(P_func(xi))
            expr_txt = str(res["expression"])
            y_d    = np.array([float(P_func(t)) for t in x_d])
            errs   = [abs(yp[i] - float(P_func(xp[i]))) for i in range(len(xp))]
            err_l2 = norme_discrete(errs, p=p_norm)
            self.ax3.plot(x_d, y_d, color=C["acc_light"], lw=2, label="Newton")

        elif method == "Moindres carrés disc":
            coeffs   = approximation_discrete(xp, yp, deg)
            P_func   = lambda t: evaluer_polynome(coeffs, t)
            val      = float(evaluer_polynome(coeffs, xi))
            termes   = [f"{c:.4f}·x^{i}" if i > 0 else f"{c:.4f}"
                        for i, c in enumerate(coeffs)]
            expr_txt = " + ".join(termes)
            y_d      = evaluer_polynome(coeffs, x_d)
            errs_sig = erreur_discrete(xp, yp, coeffs)
            errs_abs = [abs(e) for e in errs_sig]
            err_l2   = norme_discrete(errs_abs, p=p_norm)
            self.ax3.plot(x_d, y_d, color=C["acc_light"], lw=2,
                          label=f"Approx. discrète degré {deg}")

        elif method == "Moindres carrés cont":
            f_interp = lambda t: np.interp(t, xp, yp)
            coeffs   = approximation_continue(f_interp, a, b, deg)
            P_func   = lambda t: evaluer_polynome(coeffs, t)
            val      = float(evaluer_polynome(coeffs, xi))
            termes   = [f"{c:.4f}·x^{i}" if i > 0 else f"{c:.4f}"
                        for i, c in enumerate(coeffs)]
            expr_txt = " + ".join(termes)
            y_d      = evaluer_polynome(coeffs, x_d)
            # erreur continue ∫(f−P)² sur [a,b]
            err_l2   = erreur_continue(f_interp, coeffs, a, b)
            self.ax3.plot(x_d, y_d, color=C["acc_light"], lw=2,
                          label=f"Approx. continue degré {deg}")

        elif method == "Descente gradient":
            [a0, a1], historique = descente_gradient(xp, yp, taux=0.001, iterations=3000)
            coeffs   = [a0, a1]
            P_func   = lambda t: evaluer_polynome(coeffs, t)
            val      = float(evaluer_polynome(coeffs, xi))
            expr_txt = f"y = {a0:.4f} + {a1:.4f}·x"
            y_d      = evaluer_polynome(coeffs, x_d)
            errs_sig = erreur_discrete(xp, yp, coeffs)
            errs_abs = [abs(e) for e in errs_sig]
            err_l2   = norme_discrete(errs_abs, p=p_norm)
            self.ax3.plot(x_d, y_d, color=C["acc_light"], lw=2,
                          label=f"GD: y={a0:.3f}+{a1:.3f}x")

        # ── Points et point évalué ──
        self.ax3.scatter(xp, yp, color=C["amber"], zorder=5, s=55, label="Points")
        if val is not None:
            self.ax3.scatter([xi], [val], color=C["teal"], zorder=6, s=80,
                             label=f"f({xi})≈{val:.4f}")

        self.ax3.set_xlabel("x", color=C["gray"], fontsize=9)
        self.ax3.set_ylabel("y", color=C["gray"], fontsize=9)
        self.ax3.legend(facecolor=C["card"], edgecolor=C["border"],
                        labelcolor=C["white"], fontsize=8)
        self.fig3.tight_layout(pad=0.8)
        w = self.canvas3.get_tk_widget().winfo_width()
        if w > 1:
          self.fig3.set_size_inches(w / 100, 4)
        self.canvas3.draw()
        

        # ── Stats ──
        self.st3_v.set(f"{val:.6f}" if val is not None else "—")
        self.st3_m.set(method.split()[0])
        self.st3_p.set(str(len(xp)))
        self.st3_err.set(f"{err_l2:.4e}" if err_l2 is not None else "—")
        self._lbl_expr.configure(text=expr_txt)

        # ── Sauvegarde pour export ──
        self._exp_data = {
            "xp": list(xp), "yp": list(yp),
            "xi": xi, "val": val,
            "method": method, "deg": deg,
            "err_l2": err_l2,
            "expr": expr_txt,
            "a": a, "b": b,
        }

    # ──────────────────────────────────────────
    #  TABLEAU D'ERREURS
    # ──────────────────────────────────────────
    def _show_error_table(self):
        try:
            xp, yp, xi, deg, p_norm, a, b = self._parse_inputs()
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            return

        method = self.method_var.get()

        # ── Méthodes d'interpolation → tableau_erreurs_interpolation ──
        if method in ("Lagrange", "Newton diff. div."):
            polynomes = []
            if method == "Lagrange":
                res = interpolation_lagrange(xp, yp)
                polynomes.append((res["fonction"], "Lagrange"))
            else:
                res = interpolation_newton(xp, yp)
                polynomes.append((res["fonction"], "Newton"))
            # Appel de la fonction du module (console + figure matplotlib)
            plt.close('all')
            tableau_erreurs_interpolation(xp, yp, polynomes)

        # ── Méthodes d'approximation → tableau_erreurs_approximation ──
        elif method in ("Moindres carrés disc", "Moindres carrés cont",
                        "Descente gradient"):
            # On construit resultats au format attendu par tableau_erreurs_approximation :
            # { degre: {"coeffs": [...], "erreur_globale": float} }
            resultats = {}

            if method == "Moindres carrés disc":
                coeffs = approximation_discrete(xp, yp, deg)
                errs   = [abs(e) for e in erreur_discrete(xp, yp, coeffs)]
                resultats[deg] = {
                    "coeffs":        coeffs,
                    "erreur_globale": norme_discrete(errs, p=p_norm),
                }
                meilleur_degre = deg

            elif method == "Moindres carrés cont":
                f_interp = lambda t: np.interp(t, xp, yp)
                coeffs   = approximation_continue(f_interp, a, b, deg)
                err_glob = erreur_continue(f_interp, coeffs, a, b)
                resultats[deg] = {
                    "coeffs":        coeffs,
                    "erreur_globale": err_glob,
                }
                meilleur_degre = deg

            elif method == "Descente gradient":
                [a0, a1], _ = descente_gradient(xp, yp, taux=0.001, iterations=3000)
                coeffs = [a0, a1]
                errs   = [abs(e) for e in erreur_discrete(xp, yp, coeffs)]
                resultats[1] = {
                    "coeffs":        coeffs,
                    "erreur_globale": norme_discrete(errs, p=p_norm),
                }
                meilleur_degre = 1

            # Appel de la fonction du module (console + figure matplotlib)
            plt.close('all')
            tableau_erreurs_approximation(xp, yp, resultats, meilleur_degre)

    # ──────────────────────────────────────────
    #  COMPARAISON DEGRÉS 1 → 4
    # ──────────────────────────────────────────
    def _compare_degrees(self):
      try:
        xp, yp, xi, deg, p_norm, a, b = self._parse_inputs()
      except Exception as e:
        messagebox.showerror("Erreur", str(e))
        return

      method = self.method_var.get()

    # ── CAS INTERPOLATION : Lagrange vs Newton ──────────────────────────────
      if method in ("Lagrange", "Newton diff. div."):
        res_lag = interpolation_lagrange(xp, yp)
        res_new = interpolation_newton(xp, yp)

        polynomes = [
            (res_lag["fonction"], "Lagrange"),
            (res_new["fonction"], "Newton"),
        ]

        # Tracé des deux courbes côte à côte
        tracer_interpolations(xp, yp, polynomes)

        # Tableau d'erreurs point par point pour les deux méthodes
        tableau_erreurs_interpolation(xp, yp, polynomes)

    # ── CAS APPROXIMATION : comparaison degrés 1 → 4 ───────────────────────
      elif method in ("Moindres carrés disc", "Moindres carrés cont",
                    "Descente gradient"):
        degres = list(range(self.DEGRE_MIN, self.DEGRE_MAX + 1))

        meilleur_degre = None

        if method == "Moindres carrés cont":
            f_interp = lambda t: np.interp(t, xp, yp)
            resultats = {}
            for d in degres:
                try:
                    c   = approximation_continue(f_interp, a, b, d)
                    err = erreur_continue(f_interp, c, a, b)
                    resultats[d] = {"coeffs": c, "erreur_globale": err}
                except Exception:
                    pass
            if resultats:
                meilleur_degre = min(
                    resultats, key=lambda d: resultats[d]["erreur_globale"]
                )
        else:
            # Moindres carrés discrets  OU  Descente gradient
            resultats, meilleur_degre = comparer_approximations_discretes(
                xp, yp, degres, p=p_norm
            )

        if not resultats:
            messagebox.showwarning("Comparaison", "Aucun résultat disponible.")
            return

        # Tracé des courbes + segments d'erreur (meilleur en rouge)
        tracer_comparaison_approximations(xp, yp, resultats, meilleur_degre, p_norm)

        # Tableau d'erreurs détaillé
        tableau_erreurs_approximation(xp, yp, resultats, meilleur_degre)
    # ──────────────────────────────────────────
    #  NORMES
    # ──────────────────────────────────────────
    def _show_norms(self): #calcule les normes 1,2,inf pour le vecteur erreur de chaque methode 
        try:
            xp, yp, xi, deg, p_norm, a, b = self._parse_inputs()
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            return

        method = self.method_var.get()
        errs   = []

        if method == "Lagrange":
            res    = interpolation_lagrange(xp, yp)
            P_func = res["fonction"]
            errs   = [abs(yp[i] - float(P_func(xp[i]))) for i in range(len(xp))] #difference de y-P

        elif method == "Newton diff. div.":
            res    = interpolation_newton(xp, yp)
            P_func = res["fonction"]
            errs   = [abs(yp[i] - float(P_func(xp[i]))) for i in range(len(xp))]

        elif method == "Moindres carrés disc":
            coeffs = approximation_discrete(xp, yp, deg)
            errs   = [abs(e) for e in erreur_discrete(xp, yp, coeffs)]

        elif method == "Moindres carrés cont":
            f_interp = lambda t: np.interp(t, xp, yp)
            coeffs   = approximation_continue(f_interp, a, b, deg)
            # Pour les normes on évalue l'erreur aux points discrets
            errs     = [abs(yp[i] - float(evaluer_polynome(coeffs, xp[i])))
                        for i in range(len(xp))]
            # On ajoute aussi l'erreur continue L2
            err_cont = erreur_continue(f_interp, coeffs, a, b)
            n1   = norme_discrete(errs, p=1)
            n2   = norme_discrete(errs, p=2)
            ninf = norme_discrete(errs, p=inf)
            msg  = (
                f"Méthode : {method}  |  Degré : {deg}  |  [a,b] = [{a}, {b}]\n\n"
                f"Erreurs aux points discrets :\n"
                f"  {[f'{e:.4e}' for e in errs]}\n\n"
                f"Norme L1   (discret) = {n1:.6f}\n"
                f"Norme L2   (discret) = {n2:.6f}\n"
                f"Norme L∞   (discret) = {ninf:.6f}\n\n"
                f"Erreur continue ∫(f−P)²  = {err_cont:.6e}"
            )
            messagebox.showinfo("Normes des erreurs", msg)
            return

        elif method == "Descente gradient":
            [a0, a1], _ = descente_gradient(xp, yp, taux=0.001, iterations=3000)
            errs = [abs(e) for e in erreur_discrete(xp, yp, [a0, a1])]

        n1   = norme_discrete(errs, p=1)
        n2   = norme_discrete(errs, p=2)
        ninf = norme_discrete(errs, p=inf)
        msg  = (
            f"Méthode : {method}\n\n"
            f"Vecteur d'erreurs :  {[f'{e:.4e}' for e in errs]}\n\n"
            f"Norme L1   = {n1:.6f}\n"
            f"Norme L2   = {n2:.6f}\n"
            f"Norme L∞   = {ninf:.6f}"
        )
        messagebox.showinfo("Normes des erreurs", msg)

    # ──────────────────────────────────────────
    def _export3(self):
        if not self._exp_data:
            messagebox.showinfo("Info", "Aucun résultat à exporter.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="resultats_axe3.csv"
        )
        if not path:
            return
        d = self._exp_data
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["Méthode",    d["method"]])
            w.writerow(["Degré",      d.get("deg", "—")])
            w.writerow(["a",          d.get("a", "—")])
            w.writerow(["b",          d.get("b", "—")])
            w.writerow(["x évalué",   d["xi"]])
            w.writerow(["Valeur",     d["val"]])
            w.writerow(["Erreur",     d.get("err_l2", "—")])
            w.writerow(["Expression", d.get("expr", "—")])
            w.writerow([])
            w.writerow(["x_i", "y_i"])
            for xi, yi in zip(d["xp"], d["yp"]):
                w.writerow([xi, yi])
        messagebox.showinfo("Exporté", f"Sauvegardé :\n{path}")

    
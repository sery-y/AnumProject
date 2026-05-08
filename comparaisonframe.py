"""
comparaisonframe.py — Comparaison des algorithmes
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sympy as sp
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from interfacePartagee import C, F, Card, mk_entry, mk_btn, configure_treeview_style
from axe1 import dichotomie, newton_, point_fixe
from iteratives import jacobi, gauss_seidel


def calculer_ordre_point_fixe(phi_expr, solution):
    x = sp.Symbol('x')
    phi   = sp.sympify(phi_expr)
    dphi  = sp.diff(phi, x)
    ddphi = sp.diff(dphi, x)
    val_dphi  = float(dphi.subs(x, solution))
    val_ddphi = float(ddphi.subs(x, solution))
    if abs(val_dphi) > 1e-6:
        return 1, abs(val_dphi)
    if abs(val_ddphi) > 1e-6:
        return 2, abs(val_ddphi) / 2
    return 3, 0.0


class ComparaisonFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=C["bg"])
        self._build()

    def _build(self):
        top = tk.Frame(self, bg=C["bg"], padx=28, pady=18)
        top.pack(fill="x")
        tk.Label(top, text="Comparaison des algorithmes",
                 font=F["title"], bg=C["bg"], fg=C["white"]).pack(side="left")
        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

        outer = tk.Canvas(self, bg=C["bg"], bd=0, highlightthickness=0)
        self._content = tk.Frame(outer, bg=C["bg"])
        sb = ttk.Scrollbar(self, orient="vertical", command=outer.yview)
        outer.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        outer.pack(fill="both", expand=True)
        win = outer.create_window((0, 0), window=self._content, anchor="nw")
        outer.bind("<Configure>", lambda e: outer.itemconfig(win, width=e.width))
        self._content.bind("<Configure>",
                           lambda e: outer.configure(scrollregion=outer.bbox("all")))

        p = dict(padx=28, pady=8)

        # ── Type ──
        type_card = Card(self._content, "Type de problème")
        type_card.pack(fill="x", **p)
        type_row = tk.Frame(type_card.body, bg=C["card"])
        type_row.pack(fill="x")
        type_row.columnconfigure((0, 1), weight=1)

        self._type_var = tk.StringVar(value="nonlin")
        self._type_frames = {}
        for name, label, desc, col in [
            ("nonlin",   "Fonctions non linéaires", "Newton vs Dichotomie vs Point fixe", 0),
            ("lineaire", "Systèmes linéaires",       "Jacobi vs Gauss-Seidel",            1),
        ]:
            f = tk.Frame(type_row, bg=C["card2"],
                         highlightthickness=1, highlightbackground=C["border"],
                         padx=14, pady=12, cursor="hand2")
            f.grid(row=0, column=col, padx=5, pady=5, sticky="ew")
            nl = tk.Label(f, text=label, font=F["h3"], bg=C["card2"], fg=C["white"])
            nl.pack(anchor="w")
            dl = tk.Label(f, text=desc, font=F["small"], bg=C["card2"], fg=C["gray"])
            dl.pack(anchor="w")
            for w in [f, nl, dl]:
                w.bind("<Button-1>", lambda e, n=name: self._sel_type(n))
            self._type_frames[name] = f

        # ── Params non linéaires ──
        self._nl_card = Card(self._content, "Paramètres — Fonctions non linéaires")
        nl_row = tk.Frame(self._nl_card.body, bg=C["card"])
        nl_row.pack(fill="x")
        for lbl, attr, default, w in [
            ("f(x) =", "e_nl_fx",  "x**3 - x - 2", 22),
            ("a",      "e_nl_a",   "1",              5),
            ("b",      "e_nl_b",   "2",              5),
            ("x0",     "e_nl_x0",  "1.5",            5),
            ("Tol",    "e_nl_tol", "1e-6",           8),
        ]:
            col = tk.Frame(nl_row, bg=C["card"])
            col.pack(side="left", padx=(0, 10))
            tk.Label(col, text=lbl, font=F["small"],
                     bg=C["card"], fg=C["gray"]).pack(anchor="w", pady=(0, 3))
            e = mk_entry(col, default, w)
            e.pack(ipady=5)
            setattr(self, attr, e)

        # ── Params linéaires ──
        self._lin_card = Card(self._content, "Paramètres — Systèmes linéaires")
        lin_body = self._lin_card.body

        sz_row = tk.Frame(lin_body, bg=C["card"])
        sz_row.pack(fill="x", pady=(0, 8))
        tk.Label(sz_row, text="n =", font=F["small"],
                 bg=C["card"], fg=C["gray"]).pack(side="left")
        self._lin_n_var = tk.IntVar(value=3)
        for n in [2, 3, 4]:
            tk.Radiobutton(sz_row, text=str(n), variable=self._lin_n_var, value=n,
                           font=F["body"], bg=C["card"], fg=C["white"],
                           activebackground=C["card"], activeforeground=C["acc_light"],
                           selectcolor=C["acc_bg"], indicatoron=0,
                           relief="flat", bd=0, highlightthickness=1,
                           highlightbackground=C["border"],
                           cursor="hand2", padx=10, pady=3,
                           command=self._rebuild_lin_matrix).pack(side="left", padx=4)

        self._lin_mat_frame = tk.Frame(lin_body, bg=C["card"])
        self._lin_mat_frame.pack(fill="x")
        self._lin_matrix_entries = []
        self._lin_b_entries = []
        self._rebuild_lin_matrix()

        tol_row = tk.Frame(lin_body, bg=C["card"])
        tol_row.pack(fill="x", pady=(8, 0))
        for lbl, attr, default, w in [
            ("Tol",      "e_lin_tol",  "1e-6", 10),
            ("Iter max", "e_lin_nmax", "100",    6),
        ]:
            col = tk.Frame(tol_row, bg=C["card"])
            col.pack(side="left", padx=(0, 12))
            tk.Label(col, text=lbl, font=F["small"],
                     bg=C["card"], fg=C["gray"]).pack(anchor="w", pady=(0, 3))
            e = mk_entry(col, default, w)
            e.pack(ipady=5)
            setattr(self, attr, e)

        # ── Bouton ──
        btn_row = tk.Frame(self._content, bg=C["bg"], padx=28, pady=4)
        btn_row.pack(fill="x")
        mk_btn(btn_row, "  Comparer  ", self._run, color=C["accent"]).pack(side="left")

        # ── Tableau ──
        tc = Card(self._content, "Tableau comparatif")
        tc.pack(fill="x", **p)
        configure_treeview_style()
        self._tree_frame = tc.body
        self._tree = None

        # ── Graphe principal ──
        gc = Card(self._content, "Graphe de comparaison — toutes méthodes")
        gc.pack(fill="x", **p)
        self._fig = Figure(figsize=(9, 4), facecolor=C["card"])
        self._ax  = self._fig.add_subplot(111)
        self._ax.set_facecolor(C["bg"])
        self._style_ax(self._ax)
        self._canvas_plot = FigureCanvasTkAgg(self._fig, gc.body)
        self._canvas_plot.get_tk_widget().pack(fill="both")

        # ── Graphe convergence ──
        gc2 = Card(self._content, "Convergence des erreurs")
        gc2.pack(fill="x", **p)
        self._fig2 = Figure(figsize=(9, 3.2), facecolor=C["card"])
        self._ax2  = self._fig2.add_subplot(111)
        self._ax2.set_facecolor(C["bg"])
        self._style_ax(self._ax2)
        self._canvas2 = FigureCanvasTkAgg(self._fig2, gc2.body)
        self._canvas2.get_tk_widget().pack(fill="both")

        # ── Recommandation ──
        rec = tk.Frame(self._content, bg=C["teal_bg"],
                       highlightthickness=1, highlightbackground=C["teal"],
                       padx=16, pady=12)
        rec.pack(fill="x", **p)
        tk.Label(rec, text="Recommandation", font=F["h3"],
                 bg=C["teal_bg"], fg=C["teal"]).pack(anchor="w")
        self._rec = tk.Label(rec, text="Lancez une comparaison.",
                             font=F["body"], bg=C["teal_bg"], fg="#b0e8d5",
                             wraplength=900, justify="left")
        self._rec.pack(anchor="w", pady=(4, 0))

        tk.Frame(self._content, bg=C["bg"], height=20).pack()
        self._sel_type("nonlin")

    def _style_ax(self, ax):
        ax.tick_params(colors=C["gray"], labelsize=8)
        for s in ax.spines.values():
            s.set_edgecolor(C["border"])

    def _sel_type(self, name):
        self._type_var.set(name)
        for n, f in self._type_frames.items():
            is_sel = (n == name)
            bg  = C["acc_bg"] if is_sel else C["card2"]
            brd = C["accent"] if is_sel else C["border"]
            f.configure(bg=bg, highlightbackground=brd)
            for w in f.winfo_children():
                w.configure(bg=bg)
        if name == "nonlin":
            self._nl_card.pack(fill="x", padx=28, pady=8)
            self._lin_card.pack_forget()
        else:
            self._lin_card.pack(fill="x", padx=28, pady=8)
            self._nl_card.pack_forget()

    def _rebuild_lin_matrix(self):
        n = self._lin_n_var.get()
        for w in self._lin_mat_frame.winfo_children():
            w.destroy()
        self._lin_matrix_entries = []
        self._lin_b_entries = []

        hdr = tk.Frame(self._lin_mat_frame, bg=C["card"])
        hdr.pack(fill="x", pady=(0, 4))
        for j in range(n):
            tk.Label(hdr, text=f"  x{j+1}", font=F["small"],
                     bg=C["card"], fg=C["acc_light"],
                     width=7, anchor="center").grid(row=0, column=j, padx=2)
        tk.Label(hdr, text="  | b", font=F["small"],
                 bg=C["card"], fg=C["teal"],
                 width=8, anchor="center").grid(row=0, column=n, padx=2)

        default_A = [[10 if i == j else (1 if abs(i-j) == 1 else 0)
                      for j in range(n)] for i in range(n)]
        default_b = [sum(default_A[i]) for i in range(n)]

        for i in range(n):
            rf = tk.Frame(self._lin_mat_frame, bg=C["card"])
            rf.pack(fill="x", pady=2)
            row_ents = []
            for j in range(n):
                e = mk_entry(rf, str(default_A[i][j]), 7)
                e.grid(row=0, column=j, padx=2)
                row_ents.append(e)
            self._lin_matrix_entries.append(row_ents)
            tk.Label(rf, text="│", font=F["mono"],
                     bg=C["card"], fg=C["muted"]).grid(row=0, column=n, padx=4)
            be = mk_entry(rf, str(default_b[i]), 7)
            be.grid(row=0, column=n+1, padx=2)
            self._lin_b_entries.append(be)

    def _read_lin_matrix(self):
        n = self._lin_n_var.get()
        A = np.zeros((n, n))
        b = np.zeros(n)
        for i in range(n):
            for j in range(n):
                A[i, j] = float(self._lin_matrix_entries[i][j].get())
            b[i] = float(self._lin_b_entries[i].get())
        return A, b

    def _run(self):
        if self._type_var.get() == "nonlin":
            self._run_nonlin()
        else:
            self._run_lineaire()

    def _run_nonlin(self):
        f_str = self.e_nl_fx.get().strip().replace('^', '**')
        try:
            a   = float(self.e_nl_a.get())
            b   = float(self.e_nl_b.get())
            x0  = float(self.e_nl_x0.get())
            tol = float(self.e_nl_tol.get())
        except ValueError as e:
            messagebox.showerror("Erreur", str(e))
            return

        nmax  = 500
        x_sym = sp.Symbol('x')
        f_sym = sp.sympify(f_str)
        f_num = sp.lambdify(x_sym, f_sym, 'numpy')
        results = []

        # Newton
        try:
            ok, sol, iters_n, errs_n, *_ = newton_(f_str, a, b, x0, tol, nmax)
            results.append({"algo": "Newton", "ok": ok, "sol": sol,
                            "iters": iters_n, "errs": errs_n,
                            "color": C["accent"], "marker": "x", "ordre": "Quadratique"})
        except Exception:
            results.append({"algo": "Newton", "ok": False, "sol": None,
                            "iters": [], "errs": [], "color": C["accent"],
                            "marker": "x", "ordre": "—"})

        # Dichotomie
        try:
            ok, sol, iters_d, errs_d = dichotomie(f_str, a, b, tol, nmax)
            results.append({"algo": "Dichotomie", "ok": ok, "sol": sol,
                            "iters": iters_d, "errs": errs_d,
                            "color": C["teal"], "marker": "^", "ordre": "Linéaire"})
        except Exception:
            results.append({"algo": "Dichotomie", "ok": False, "sol": None,
                            "iters": [], "errs": [], "color": C["teal"],
                            "marker": "^", "ordre": "—"})

        # Point fixe
        try:
            ok, phi, sol, iters_pf, errs_pf, rapport = point_fixe(f_str, a, b, x0, tol, nmax)
            if ok and sol is not None and phi is not None:
                try:
                    ordre, taux = calculer_ordre_point_fixe(str(phi), sol)
                    ordre_str = f"Ordre {ordre} (taux={taux:.4f})"
                except Exception:
                    ordre_str = "Linéaire"
            else:
                ordre_str = "—"
            results.append({"algo": "Point fixe", "ok": ok, "sol": sol,
                            "iters": iters_pf, "errs": errs_pf,
                            "color": C["amber"], "marker": "o", "ordre": ordre_str})
        except Exception:
            results.append({"algo": "Point fixe", "ok": False, "sol": None,
                            "iters": [], "errs": [], "color": C["amber"],
                            "marker": "o", "ordre": "—"})

        self._build_tree_nonlin(results)

        # Graphe principal : f(x) + points de chaque méthode
        self._ax.cla()
        self._style_ax(self._ax)
        try:
            xs = np.linspace(a, b, 500)
            ys = np.clip(f_num(xs), -50, 50)
            self._ax.plot(xs, ys, color=C["white"], lw=1.5, label="f(x)", zorder=1)
            self._ax.axhline(0, color=C["muted"], lw=0.7, ls="--")
            for r in results:
                if r["ok"] and r["iters"]:
                    xi = r["iters"]
                    yi = [float(np.clip(f_num(v), -50, 50)) for v in xi]
                    self._ax.scatter(xi, yi, color=r["color"], marker=r["marker"],
                                     s=40, zorder=3, label=r["algo"], alpha=0.85)
                    if r["sol"] is not None:
                        self._ax.axvline(r["sol"], color=r["color"],
                                         lw=1, ls=":", alpha=0.5)
            self._ax.set_xlabel("x", color=C["gray"], fontsize=9)
            self._ax.set_ylabel("f(x)", color=C["gray"], fontsize=9)
            self._ax.legend(facecolor=C["card"], edgecolor=C["border"],
                            labelcolor=C["white"], fontsize=8)
        except Exception:
            pass
        self._fig.tight_layout(pad=0.8)
        self._canvas_plot.draw()

        # Graphe convergence erreurs
        self._ax2.cla()
        self._style_ax(self._ax2)
        has = False
        for r in results:
            if r["ok"] and r["errs"] and len(r["errs"]) > 1:
                errs_pos = [max(e, 1e-16) for e in r["errs"]]
                self._ax2.semilogy(range(len(errs_pos)), errs_pos,
                                   color=r["color"], lw=2, marker="o",
                                   markersize=3, label=r["algo"])
                has = True
        if has:
            self._ax2.set_xlabel("Itération", color=C["gray"], fontsize=9)
            self._ax2.set_ylabel("Erreur", color=C["gray"], fontsize=9)
            self._ax2.legend(facecolor=C["card"], edgecolor=C["border"],
                             labelcolor=C["white"], fontsize=8)
        self._fig2.tight_layout(pad=0.8)
        self._canvas2.draw()

        self._rec.configure(
            text="• Newton : rapide (quadratique) mais nécessite f dérivable et bon x0\n"
                 "• Dichotomie : lente (linéaire) mais toujours stable si f(a)·f(b) < 0\n"
                 "• Point fixe : simple mais dépend du choix de φ(x)\n"
                 "→ Recommandé : Newton si f dérivable, Dichotomie sinon"
        )

    def _run_lineaire(self):
        try:
            A, b = self._read_lin_matrix()
            tol  = float(self.e_lin_tol.get())
            nmax = int(self.e_lin_nmax.get())
        except ValueError as e:
            messagebox.showerror("Erreur", str(e))
            return

        x0 = np.zeros(len(b))
        results = []

        # Jacobi
        try:
            conv, x_sol, hist_x, errs, rho_j, Js, M_norm = jacobi(
                A.copy(), b.copy(), x0, tol, nmax)
            res = float(np.linalg.norm(A @ x_sol - b)) if x_sol is not None else float('inf')
            results.append({"algo": "Jacobi", "ok": conv, "sol": x_sol,
                            "iters": len(hist_x), "err": res, "rho": rho_j,
                            "errs": errs, "color": C["accent"]})
        except Exception:
            results.append({"algo": "Jacobi", "ok": False, "sol": None,
                            "iters": 0, "err": float('inf'), "rho": None,
                            "errs": [], "color": C["accent"]})

        # Gauss-Seidel
        try:
            conv, x_sol, hist_x, errs, rho_gs, Gs, M_norm = gauss_seidel(
                A.copy(), b.copy(), x0, tol, nmax)
            res = float(np.linalg.norm(A @ x_sol - b)) if x_sol is not None else float('inf')
            results.append({"algo": "Gauss-Seidel", "ok": conv, "sol": x_sol,
                            "iters": len(hist_x), "err": res, "rho": rho_gs,
                            "errs": errs, "color": C["teal"]})
        except Exception:
            results.append({"algo": "Gauss-Seidel", "ok": False, "sol": None,
                            "iters": 0, "err": float('inf'), "rho": None,
                            "errs": [], "color": C["teal"]})

        self._build_tree_lineaire(results)

        # Graphe barres itérations + résidu
        self._ax.cla()
        self._style_ax(self._ax)
        algos_ok = [r for r in results if r["ok"]]
        if algos_ok:
            names = [r["algo"] for r in algos_ok]
            iters = [r["iters"] for r in algos_ok]
            errs  = [r["err"]   for r in algos_ok]
            x_pos = np.arange(len(names))
            w     = 0.35
            self._ax.bar(x_pos - w/2, iters, w,
                         color=[C["accent"], C["teal"]][:len(algos_ok)],
                         alpha=0.85, label="Itérations")
            ax2 = self._ax.twinx()
            ax2.bar(x_pos + w/2, errs, w,
                    color=[C["amber"], C["acc_light"]][:len(algos_ok)],
                    alpha=0.85, label="Résidu ‖Ax-b‖")
            self._ax.set_xticks(x_pos)
            self._ax.set_xticklabels(names, color=C["gray"], fontsize=9)
            self._ax.set_ylabel("Itérations", color=C["gray"], fontsize=8)
            ax2.set_ylabel("Résidu", color=C["gray"], fontsize=8)
            ax2.tick_params(colors=C["gray"], labelsize=8)
            for s in ax2.spines.values():
                s.set_edgecolor(C["border"])
            l1, lab1 = self._ax.get_legend_handles_labels()
            l2, lab2 = ax2.get_legend_handles_labels()
            self._ax.legend(l1 + l2, lab1 + lab2,
                            facecolor=C["card"], edgecolor=C["border"],
                            labelcolor=C["white"], fontsize=8)
        self._fig.tight_layout(pad=0.8)
        self._canvas_plot.draw()

        # Graphe convergence
        self._ax2.cla()
        self._style_ax(self._ax2)
        has = False
        for r in results:
            if r["errs"] and len(r["errs"]) > 1:
                errs_pos = [max(e, 1e-16) for e in r["errs"]]
                self._ax2.semilogy(range(len(errs_pos)), errs_pos,
                                   color=r["color"], lw=2, marker="o",
                                   markersize=3, label=r["algo"])
                has = True
        if has:
            self._ax2.set_xlabel("Itération", color=C["gray"], fontsize=9)
            self._ax2.set_ylabel("Erreur estimée", color=C["gray"], fontsize=9)
            self._ax2.legend(facecolor=C["card"], edgecolor=C["border"],
                             labelcolor=C["white"], fontsize=8)
        self._fig2.tight_layout(pad=0.8)
        self._canvas2.draw()

        # Recommandation
        j_ok  = next((r for r in results if r["algo"] == "Jacobi"       and r["ok"]), None)
        gs_ok = next((r for r in results if r["algo"] == "Gauss-Seidel" and r["ok"]), None)
        if j_ok and gs_ok:
            faster      = "Jacobi" if j_ok["iters"] <= gs_ok["iters"] else "Gauss-Seidel"
            more_precise = "Jacobi" if j_ok["err"]   <= gs_ok["err"]   else "Gauss-Seidel"
            rec = (
                f"• Jacobi       : ρ={j_ok['rho']:.4f}, {j_ok['iters']} itérations, résidu={j_ok['err']:.2e}\n"
                f"• Gauss-Seidel : ρ={gs_ok['rho']:.4f}, {gs_ok['iters']} itérations, résidu={gs_ok['err']:.2e}\n"
                f"→ Plus rapide   : {faster}\n"
                f"→ Plus précis   : {more_precise}\n"
                f"→ Gauss-Seidel converge généralement plus vite si ρ(Gs) < ρ(Js)"
            )
        elif gs_ok:
            rec = "Jacobi n'a pas convergé. Gauss-Seidel recommandé."
        elif j_ok:
            rec = "Gauss-Seidel n'a pas convergé. Jacobi recommandé."
        else:
            rec = "Aucune méthode n'a convergé — vérifiez que la matrice est diagonalement dominante."
        self._rec.configure(text=rec)

    def _build_tree_nonlin(self, results):
        if self._tree:
            self._tree.destroy()
        cols = ("algo", "ok", "sol", "iters", "err_fin", "ordre")
        hdrs = [("algo",    "Algorithme",   140), ("ok",      "Succès",       60),
                ("sol",     "Solution",     120), ("iters",   "Itérations",   90),
                ("err_fin", "Erreur finale",110), ("ordre",   "Convergence", 160)]
        tree = ttk.Treeview(self._tree_frame, columns=cols,
                            show="headings", height=len(results), style="D.Treeview")
        for col, title, w in hdrs:
            tree.heading(col, text=title)
            tree.column(col, width=w, anchor="center")
        for r in results:
            tree.insert("", "end", values=(
                r["algo"],
                "✓" if r["ok"] else "✗",
                f"{r['sol']:.8f}" if r["sol"] is not None else "—",
                str(len(r["iters"])) if r["iters"] else "0",
                f"{r['errs'][-1]:.2e}" if r["errs"] else "—",
                r["ordre"]
            ), tags=("ok" if r["ok"] else "fail",))
        tree.tag_configure("ok",   foreground=C["white"])
        tree.tag_configure("fail", foreground=C["muted"])
        tree.pack(fill="x")
        self._tree = tree

    def _build_tree_lineaire(self, results):
        if self._tree:
            self._tree.destroy()
        cols = ("algo", "ok", "iters", "err", "rho")
        hdrs = [("algo",  "Algorithme",    160), ("ok",    "Convergé",      70),
                ("iters", "Itérations",     90), ("err",   "Résidu ‖Ax-b‖",130),
                ("rho",   "Rayon spectral", 120)]
        tree = ttk.Treeview(self._tree_frame, columns=cols,
                            show="headings", height=len(results), style="D.Treeview")
        for col, title, w in hdrs:
            tree.heading(col, text=title)
            tree.column(col, width=w, anchor="center")
        for r in results:
            tree.insert("", "end", values=(
                r["algo"],
                "✓" if r["ok"] else "✗",
                r["iters"],
                f"{r['err']:.2e}" if r["err"] != float('inf') else "∞",
                f"{r['rho']:.4f}" if r["rho"] is not None else "—"
            ), tags=("ok" if r["ok"] else "fail",))
        tree.tag_configure("ok",   foreground=C["white"])
        tree.tag_configure("fail", foreground=C["muted"])
        tree.pack(fill="x")
        self._tree = tree
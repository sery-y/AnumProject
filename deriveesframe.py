
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sympy as sp
import csv

from interfacePartagee import C, F, Card, StatCard, mk_entry, mk_btn, configure_treeview_style


class DeriveesFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=C["bg"])
        self._build()

    def _build(self):
        # ── Top bar ──
        top = tk.Frame(self, bg=C["bg"], padx=28, pady=18)
        top.pack(fill="x")
        tk.Label(top, text="Étude de fonction — Dérivées & Continuité",
                 font=F["title"], bg=C["bg"], fg=C["white"]).pack(side="left")
        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

        # ── Scrollable canvas ──
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

        # ── Saisie fonction ──
        fc = Card(self._content, "Fonction à étudier")
        fc.pack(fill="x", **p)
        row = tk.Frame(fc.body, bg=C["card"])
        row.pack(fill="x")

        for lbl, attr, default, w in [
            ("f(x) =", "e_fx", "x**3 - 3*x + 2", 28),
            ("a (début)", "e_a", "-3", 6),
            ("b (fin)",   "e_b", "3",  6),
        ]:
            col = tk.Frame(row, bg=C["card"])
            col.pack(side="left", padx=(0, 14))
            tk.Label(col, text=lbl, font=F["small"],
                     bg=C["card"], fg=C["gray"]).pack(anchor="w", pady=(0, 3))
            e = mk_entry(col, default, w)
            e.pack(ipady=5)
            setattr(self, attr, e)

        col_b = tk.Frame(row, bg=C["card"])
        col_b.pack(side="left")
        tk.Label(col_b, text=" ", font=F["small"], bg=C["card"], fg=C["gray"]).pack(pady=(0, 3))
        mk_btn(col_b, "  Analyser  ", self._run, color=C["accent"]).pack(ipady=4)

        # ── Résultats symboliques ──
        self._sym_card = Card(self._content, "Résultats symboliques")
        self._sym_card.pack(fill="x", **p)
        self._sym_body = self._sym_card.body

        self._sym_labels = {}
        for key, label in [
            ("f",    "f(x)"),
            ("fp",   "f'(x)  — Dérivée 1ère"),
            ("fpp",  "f''(x) — Dérivée 2ème"),
            ("fppp", "f'''(x) — Dérivée 3ème"),
    
            ("lim_inf", "lim f(x) quand x → -∞"),
            ("lim_sup", "lim f(x) quand x → +∞"),
        ]:
            row2 = tk.Frame(self._sym_body, bg=C["card"])
            row2.pack(fill="x", pady=2)
            tk.Label(row2, text=label, font=F["small"],
                     bg=C["card"], fg=C["gray"], width=28, anchor="w").pack(side="left")
            lbl = tk.Label(row2, text="—", font=F["mono"],
                           bg=C["card"], fg=C["acc_light"], anchor="w")
            lbl.pack(side="left", fill="x", expand=True)
            self._sym_labels[key] = lbl
          

        # ── Points remarquables ──
        pc = Card(self._content, "Points remarquables")
        pc.pack(fill="x", **p)
        self._pts_labels = {}
        col2 = tk.Frame(pc.body, bg=C["card2"],
                        highlightthickness=1, highlightbackground=C["border"],
                        padx=12, pady=10)
        col2.pack(side="left", fill="x", expand=True)
        tk.Label(col2, text="Zéros f(x) = 0", font=F["small"],
                 bg=C["card2"], fg=C["gray"]).pack(anchor="w")
        lbl = tk.Label(col2, text="—", font=F["mono"],
                       bg=C["card2"], fg=C["teal"], wraplength=400, justify="left")
        lbl.pack(anchor="w")
        self._pts_labels["zeros"] = lbl


        # ── Continuité & Domaine ──
        cc = Card(self._content, "Continuité & Domaine")
        cc.pack(fill="x", **p)
        self._cont_body = cc.body

        cont_row = tk.Frame(self._cont_body, bg=C["card"])
        cont_row.pack(fill="x")
        self._cont_labels = {}
        for col_i, (key, label) in enumerate([
            
            ("continuite",  "Continue sur [a,b]"),
            ("derivable",   "Dérivable sur [a,b]"),
           
        ]):
            f = tk.Frame(cont_row, bg=C["card2"],
                         highlightthickness=1, highlightbackground=C["border"],
                         padx=12, pady=10)
            f.pack(side="left", padx=(0, 8), fill="x", expand=True)
            tk.Label(f, text=label, font=F["small"],
                     bg=C["card2"], fg=C["gray"]).pack(anchor="w")
            lbl = tk.Label(f, text="—", font=F["h3"],
                           bg=C["card2"], fg=C["muted"])
            lbl.pack(anchor="w")
            self._cont_labels[key] = lbl

       

        # ── Tableau de variation ──
        tv_card = Card(self._content, "Tableau de variation")
        tv_card.pack(fill="x", **p)
        self._tv_frame = tv_card.body
        self._tv_canvas = tk.Canvas(self._tv_frame, bg=C["card"],
                                    highlightthickness=0, height=160)
        self._tv_canvas.pack(fill="x")

        # ── Graphe ──
        gc = Card(self._content, "Visualisation graphique")
        gc.pack(fill="x", **p)
        self._fig = Figure(figsize=(9, 3.5), facecolor=C["card"])
        self._ax  = self._fig.add_subplot(111)
        self._ax.set_facecolor(C["bg"])
        self._style_ax(self._ax)
        self._canvas_plot = FigureCanvasTkAgg(self._fig, gc.body)
        self._canvas_plot.get_tk_widget().pack(fill="both")

        

        tk.Frame(self._content, bg=C["bg"], height=20).pack()

    # ──────────────────────────────────────────
    def _style_ax(self, ax):
        ax.tick_params(colors=C["gray"], labelsize=8)
        for sp_ in ax.spines.values():
            sp_.set_edgecolor(C["border"])
        ax.set_xlabel("x", color=C["gray"], fontsize=9)
        ax.set_ylabel("f(x)", color=C["gray"], fontsize=9)

    # ──────────────────────────────────────────
    def _run(self):
      f_str = self.e_fx.get().strip().replace('^', '**')
      try:
        a = float(self.e_a.get())
        b = float(self.e_b.get())
      except ValueError:
        messagebox.showerror("Erreur", "a et b doivent être des nombres.")
        return

      x = sp.Symbol('x')
      try:
        f_sym = sp.sympify(f_str)
      except Exception as e:
        messagebox.showerror("Erreur", f"Fonction invalide : {e}")
        return

    # ── Dérivées & primitive ──
      fp_sym   = sp.diff(f_sym, x)
      fpp_sym  = sp.diff(fp_sym, x)
      fppp_sym = sp.diff(fpp_sym, x)
      

      self._sym_labels["f"].configure(text=str(f_sym))
      self._sym_labels["fp"].configure(text=str(fp_sym))
      self._sym_labels["fpp"].configure(text=str(fpp_sym))
      self._sym_labels["fppp"].configure(text=str(fppp_sym))
      
    # ── Limites ──
      try:
        lim_m = sp.limit(f_sym, x, -sp.oo)
        lim_p = sp.limit(f_sym, x, sp.oo)
        self._sym_labels["lim_inf"].configure(text=str(lim_m))
        self._sym_labels["lim_sup"].configure(text=str(lim_p))
      except Exception:
        self._sym_labels["lim_inf"].configure(text="indéterminée")
        self._sym_labels["lim_sup"].configure(text="indéterminée")

      # ── Zéros de f' (pour tableau de variation uniquement) ──
      try:
        crits = sp.solve(fp_sym, x)
        crits_real = [c for c in crits if sp.im(c) == 0]
        crits_in   = [c for c in crits_real if a <= float(c.evalf()) <= b]
      except Exception:
        crits_in = []

    # ── Continuité ──
      try:
        f_num = sp.lambdify(x, f_sym, 'numpy')
        xs_test = np.linspace(a, b, 500)
        ys_test = f_num(xs_test)
        is_cont = np.all(np.isfinite(ys_test))
        sing = sp.singularities(f_sym, x)
        sing_in = [s for s in sing
           if sp.im(s) == 0 and a <= float(s.evalf()) <= b]
        if sing_in:
          is_cont = False
        self._cont_labels["continuite"].configure(
            text=" Oui" if is_cont else " Non",
            fg=C["teal"] if is_cont else C["red"])
      except Exception:
        self._cont_labels["continuite"].configure(text="? Indéterminé", fg=C["muted"])

    # ── Dérivabilité ──
    #verification de f prime finie
      try:
        fp_num = sp.lambdify(x, fp_sym, 'numpy')
        ys_fp  = fp_num(xs_test)
        is_der = np.all(np.isfinite(ys_fp))
        self._cont_labels["derivable"].configure(
            text=" Oui" if is_der else " Non",
            fg=C["teal"] if is_der else C["red"])
        #verification si f admet des points de discontinuité( ou f n est pas definie) pas continue-> pas derivable !
        try:
          sing = sp.singularities(f_sym, x)
          sing_in = [s for s in sing 
                   if sp.im(s) == 0 and a <= float(s.evalf()) <= b]
          if sing_in:
            is_der = False
        except Exception:
           pass
        #calculer la derivee gauche et droite dans les points critique 0 et extremités
        points_to_check = crits_in + [sp.Float(a), sp.Float(b)]
        for pt in points_to_check:
          try:
            lim_g = sp.limit(fp_sym, x, pt, '-')
            lim_d = sp.limit(fp_sym, x, pt, '+')
            if lim_g in (sp.oo, -sp.oo) or lim_d in (sp.oo, -sp.oo): #si les limite sont infinies pas derivable
              is_der = False
            elif sp.simplify(lim_g - lim_d) != 0:
                is_der = False
          except Exception:
            pass

        self._cont_labels["derivable"].configure(
          text=" Oui" if is_der else " Non",
          fg=C["teal"] if is_der else C["red"])
      except Exception as e:
        print(f"Erreur dérivabilité : {type(e).__name__}: {e}")
        self._cont_labels["derivable"].configure(text="? Indéterminé", fg=C["muted"])

    # ── Zéros de f ──
      try:
        zeros = sp.solve(f_sym, x)
        zeros_real = [z for z in zeros if sp.im(z) == 0]
        zeros_in   = [z for z in zeros_real if a <= float(z.evalf()) <= b]
        txt_z = ", ".join([f"x={float(z.evalf()):.4f}" for z in zeros_in]) or "Aucun sur [a,b]"
        self._pts_labels["zeros"].configure(text=txt_z)
      except Exception:
        zeros_in = []
        self._pts_labels["zeros"].configure(text="Non calculable")

    

    # ── Tableau de variation ──
      self._draw_variation_table(f_sym, fp_sym, a, b, crits_in)

    # ── Graphe ──
      self._ax.cla()
      self._style_ax(self._ax)
      try:
        f_num  = sp.lambdify(x, f_sym,  'numpy')
        fp_num = sp.lambdify(x, fp_sym, 'numpy')
        xs = np.linspace(a, b, 600)
        ys = np.array(f_num(xs),  dtype=float)
        yp = np.array(fp_num(xs), dtype=float)
        ys = np.where(np.abs(ys) > 1e6, np.nan, ys)
        yp = np.where(np.abs(yp) > 1e6, np.nan, yp)
        self._ax.plot(xs, ys, color=C["acc_light"], lw=2, label="f(x)")
        self._ax.plot(xs, yp, color=C["teal"], lw=1.5, ls="--", alpha=0.8, label="f'(x)")
        self._ax.axhline(0, color=C["muted"], lw=0.7, ls="--")
        self._ax.axvline(0, color=C["muted"], lw=0.7, ls="--")
        if zeros_in:
            xz = [float(z.evalf()) for z in zeros_in]
            self._ax.scatter(xz, [0]*len(xz), color=C["teal"], s=60, zorder=6, label="Zéros")
        self._ax.legend(facecolor=C["card"], edgecolor=C["border"],
                        labelcolor=C["white"], fontsize=8)
      except Exception:
        pass
      self._fig.tight_layout(pad=0.8)
      self._canvas_plot.draw()

      self._last_data = {"f": str(f_sym), "fp": str(fp_sym), "fpp": str(fpp_sym), "a": a, "b": b}


    def _draw_variation_table(self, f_sym, fp_sym, a, b, crits_in):
      canvas = self._tv_canvas
      canvas.delete("all")

      x = sp.Symbol('x')

    # Points clés : a + zéros de f' dans [a,b] + b
      crits_vals = []
      for c in crits_in:
        try:
            val = float(c.evalf())
            if a <= val <= b:
                crits_vals.append(round(val, 8))
        except Exception:
            pass

      pts = sorted(set([round(a, 8)] + crits_vals + [round(b, 8)]))
      n = len(pts)

      W       = 900
      H       = 160
      label_w = 80
      avail   = W - label_w
      col_w   = avail // (n - 1) if n > 1 else avail

      brd  = C["border"]
      gray = C["gray"]
      wht  = C["white"]
      teal = C["teal"]
      acc  = C["acc_light"]
      red  = C["red"]

      canvas.configure(height=H)

    # ── Lignes horizontales ──
      for y in [0, 40, 80, 120, 160]:
        canvas.create_line(0, y, W, y, fill=brd)

    # ── Ligne verticale labels ──
      canvas.create_line(label_w, 0, label_w, H, fill=brd)

    # ── Labels des lignes ──
      for y, txt in [(20, "x"), (60, "f'(x)"), (100, "Variation"), (140, "f(x)")]:
        canvas.create_text(label_w // 2, y, text=txt, fill=gray,
                           font=("Consolas", 9, "bold"))

    # ── Colonnes pour chaque point ──
      for i, pt in enumerate(pts):
        if i == n - 1:
            cx = label_w + (n - 1) * col_w
        else:
            cx = label_w + i * col_w

        # ligne verticale séparatrice
        if i > 0:
            canvas.create_line(label_w + i * col_w, 0,
                               label_w + i * col_w, H, fill=brd)

        # valeur de x
        canvas.create_text(cx, 20, text=f"{pt:.3g}",
                           fill=wht, font=("Consolas", 9))

        # signe de f'(x) au point
        try:
            fp_val = float(fp_sym.subs(x, pt).evalf())
            if abs(fp_val) < 1e-8:
                canvas.create_text(cx, 60, text="0",
                                   fill=C["amber"], font=("Consolas", 10, "bold"))
            elif fp_val > 0:
                canvas.create_text(cx, 60, text="+",
                                   fill=teal, font=("Consolas", 12, "bold"))
            else:
                canvas.create_text(cx, 60, text="−",
                                   fill=red, font=("Consolas", 12, "bold"))
        except Exception:
            canvas.create_text(cx, 60, text="?", fill=gray, font=("Consolas", 9))

        # valeur de f(x) au point
        try:
            fv = float(f_sym.subs(x, pt).evalf())
            canvas.create_text(cx, 140, text=f"{fv:.3g}",
                               fill=acc, font=("Consolas", 9))
        except Exception:
            canvas.create_text(cx, 140, text="?", fill=gray, font=("Consolas", 9))

    # ── Variation entre deux points consécutifs ──
      for i in range(n - 1):
        x_mid   = (pts[i] + pts[i + 1]) / 2
        x_center = label_w + i * col_w + col_w // 2
        try:
            fp_mid = float(fp_sym.subs(x, x_mid).evalf())
            if fp_mid > 0:
                canvas.create_text(x_center, 100, text="Croissante",
                                   fill=teal, font=("Consolas", 9, "bold"))
            else:
                canvas.create_text(x_center, 100, text="Décroissante",
                                   fill=red, font=("Consolas", 9, "bold"))
        except Exception:
            pass
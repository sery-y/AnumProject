import tkinter as tk
from tkinter import ttk

from interfacePartagee import C, F
from axe1frame import Axe1Frame
from axe2frame import Axe2Frame
from axe3frame import Axe3Frame


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Analyse Numérique — USTHB 3INGSoft")
        self.geometry("1280x800")
        self.minsize(1000, 650)
        self.configure(bg=C["bg"])
        self._build()

    # ──────────────────────────────────────────
    def _build(self):
        # ── Sidebar ──
        sidebar = tk.Frame(self, bg=C["panel"], width=230)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        logo = tk.Frame(sidebar, bg=C["panel"], padx=18, pady=22)
        logo.pack(fill="x")
        tk.Label(logo, text="Analyse\nnumérique", font=("Segoe UI", 13, "bold"),
                 bg=C["panel"], fg=C["white"], justify="left").pack(anchor="w")
        tk.Frame(sidebar, bg=C["border"], height=1).pack(fill="x")

        nav = tk.Frame(sidebar, bg=C["panel"], pady=14)
        nav.pack(fill="x")
        tk.Label(nav, text="AXES", font=("Segoe UI", 8, "bold"),
                 bg=C["panel"], fg=C["muted"], padx=18).pack(anchor="w", pady=(0, 4))

        self._nav_btns = {}
        nav_items = [
            ("axe1", "  Fonctions non linéaires", C["accent"]),
            ("axe2", "  Systèmes linéaires",       C["teal"]),
            ("axe3", "  Interpolation & Approx.",  C["amber"]),
        ]
        for key, text, color in nav_items:
            row = tk.Frame(nav, bg=C["panel"], cursor="hand2")
            row.pack(fill="x")
            dot = tk.Label(row, text="●", font=("Segoe UI", 8),
                           bg=C["panel"], fg=color, padx=18, pady=10)
            dot.pack(side="left")
            lbl = tk.Label(row, text=text, font=F["body"],
                           bg=C["panel"], fg=C["gray"], pady=10, anchor="w")
            lbl.pack(side="left", fill="x", expand=True)
            for w in [row, dot, lbl]:
                w.bind("<Button-1>", lambda e, k=key: self._switch(k))
                w.bind("<Enter>",  lambda e, r=row, d=dot, l=lbl: [
                    r.configure(bg=C["hover"]),
                    d.configure(bg=C["hover"]),
                    l.configure(bg=C["hover"], fg=C["white"])])
                w.bind("<Leave>",  lambda e, k=key, r=row, d=dot, l=lbl:
                       self._nav_leave(k, r, d, l))
            self._nav_btns[key] = (row, dot, lbl, color)

        tk.Frame(sidebar, bg=C["border"], height=1).pack(fill="x")

        nav2 = tk.Frame(sidebar, bg=C["panel"], pady=14)
        nav2.pack(fill="x")
        tk.Label(nav2, text="OUTILS", font=("Segoe UI", 8, "bold"),
                 bg=C["panel"], fg=C["muted"], padx=18).pack(anchor="w", pady=(0, 4))
        for txt in ["d  Dérivées & continuité", "=  Comparaison algos"]:
            l = tk.Label(nav2, text=txt, font=F["body"],
                         bg=C["panel"], fg=C["gray"], padx=18, pady=8,
                         anchor="w", cursor="hand2")
            l.pack(fill="x")
            l.bind("<Enter>", lambda e, b=l: b.configure(bg=C["hover"], fg=C["white"]))
            l.bind("<Leave>", lambda e, b=l: b.configure(bg=C["panel"], fg=C["gray"]))

        # ── Zone principale ──
        self._main = tk.Frame(self, bg=C["bg"])
        self._main.pack(side="left", fill="both", expand=True)
        self._frames = {
            "axe1": Axe1Frame(self._main),
            "axe2": Axe2Frame(self._main),
            "axe3": Axe3Frame(self._main),
        }
        self._current = None
        self._switch("axe1")

    # ──────────────────────────────────────────
    def _switch(self, key):
        if self._current:
            self._frames[self._current].pack_forget()
        self._frames[key].pack(fill="both", expand=True)
        self._current = key
        for k, (row, dot, lbl, color) in self._nav_btns.items():
            if k == key:
                row.configure(bg=C["hover"])
                dot.configure(bg=C["hover"])
                lbl.configure(bg=C["hover"], fg=C["white"])
            else:
                row.configure(bg=C["panel"])
                dot.configure(bg=C["panel"])
                lbl.configure(bg=C["panel"], fg=C["gray"])

    def _nav_leave(self, key, row, dot, lbl):
        if self._current == key:
            row.configure(bg=C["hover"])
            dot.configure(bg=C["hover"])
            lbl.configure(bg=C["hover"], fg=C["white"])
        else:
            row.configure(bg=C["panel"])
            dot.configure(bg=C["panel"])
            lbl.configure(bg=C["panel"], fg=C["gray"])


if __name__ == "__main__":
    app = App()
    app.mainloop()
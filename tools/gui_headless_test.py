import sys
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tkinter as tk
from types import SimpleNamespace

from gui.gui import ArenaGUI


def run():
    root = tk.Tk()
    # don't map window too large; we won't call mainloop - just use the app logic
    app = ArenaGUI(root)

    # simulate a left-click event at tile (8,23)
    ev = SimpleNamespace(x=8 * app.tile_size + 2, y=23 * app.tile_size + 2)
    app.on_left_click(ev)
    print('Deployed units:', len(app.engine.units))
    # run update_loop iterations manually, sleeping between to emulate GUI frame timing
    for i in range(60):
        app.update_loop()
        if app.engine.units:
            u = app.engine.units[0]
            print(f'iter {i:02d}: time={app.engine.time:.2f} pos=({u.x:.3f},{u.y:.3f})')
        else:
            print(f'iter {i:02d}: no units')
        time.sleep(0.033)

    root.destroy()

if __name__ == '__main__':
    run()

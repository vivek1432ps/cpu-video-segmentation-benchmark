"""Desktop prompt-to-mask lab: open an image, enter labels, and export masks."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import strftime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import ImageTk
from prompt_segment import Segmenter, load_image, parse_prompts, save_result

BG, PANEL, CYAN, PINK, FG, DIM = (
    "#090714", "#18112a", "#35f7ff", "#ff61c4", "#eef4fa", "#c0bbd6"
)


class PromptLab:
    def __init__(self, root):
        self.root = root
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.segmenter = self.result = None
        self.image = self.path = None
        self.busy = False
        self.photos = {}
        root.title("CPU Segmentation Lab | YOLOE-26")
        root.geometry("1000x685")
        root.minsize(720, 620)
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.close)
        theme = ttk.Style(root)
        theme.theme_use("clam")
        theme.configure("TButton", background=CYAN, foreground=BG, padding=(13, 7))
        theme.map("TButton", background=[("disabled", "#433954"), ("active", "#9bfcff")],
                  foreground=[("disabled", DIM)])
        theme.configure("TEntry", fieldbackground=PANEL, foreground=FG, insertcolor=CYAN)
        parent = tk.Frame(root, bg=BG, padx=24, pady=20)
        parent.pack(fill="both", expand=True)
        self.label(parent, "LOCAL CPU DEMO  /  YOLOE-26n", CYAN, 10).pack(anchor="w")
        self.label(parent, "Segment what you want", FG, 26).pack(anchor="w", pady=(4, 4))
        self.label(parent, "Enter object names to generate instance masks without retraining.",
                   DIM, 11).pack(anchor="w", pady=(0, 16))

        controls = tk.Frame(parent, bg=PANEL, padx=12, pady=12)
        controls.pack(fill="x")
        self.pick = ttk.Button(controls, text="Open image…", command=self.open_image)
        self.pick.pack(side="left", padx=(0, 10))
        controls.columnconfigure(0, weight=1)
        self.prompt = tk.StringVar(value="person, bus")
        self.entry = ttk.Entry(controls, textvariable=self.prompt)
        self.entry.pack(side="left", expand=True, fill="x", ipady=4, padx=(0, 10))
        self.entry.bind("<Return>", lambda _event: self.run())
        self.segment = ttk.Button(controls, text="Segment", command=self.run, state="disabled")
        self.segment.pack(side="left", padx=(0, 8))
        self.export = ttk.Button(controls, text="Export masks…", command=self.save, state="disabled")
        self.export.pack(side="left")

        panels = tk.Frame(parent, bg=BG)
        panels.pack(fill="both", expand=True, pady=16)
        panels.rowconfigure(0, weight=1)
        for i, (key, title) in enumerate((("input", "01 / INPUT"), ("output", "02 / MASK OVERLAY"))):
            panels.columnconfigure(i, weight=1, uniform="preview")
            card = tk.Frame(panels, bg=PANEL, padx=12, pady=12)
            card.grid(row=0, column=i, sticky="nsew", padx=(0, 6) if i == 0 else (6, 0))
            self.label(card, title, CYAN if i == 0 else PINK, 10, PANEL).pack(anchor="w")
            preview = tk.Label(card, bg=PANEL, fg=DIM,
                               text="Open an image to begin" if i == 0 else "No prediction yet")
            preview.pack(fill="both", expand=True)
            setattr(self, key, preview)
            preview.bind("<Configure>", lambda event, k=key: self.resize(k, event))

        self.status = self.label(parent, "First run downloads the model and text encoder.", DIM, 10)
        self.status.pack(anchor="w", fill="x")

    @staticmethod
    def label(parent, text, fg, size, bg=BG):
        return tk.Label(parent, text=text, fg=fg, bg=bg, font=("TkDefaultFont", size),
                        justify="left", anchor="w", wraplength=890)

    def resize(self, key, event):
        if event.width > 20 and event.height > 20:
            self.preview(key, event.width - 10, event.height - 10)

    def preview(self, key, width=None, height=None):
        widget = getattr(self, key)
        image = self.image if key == "input" else (self.result[0] if self.result else None)
        if image is None:
            widget.configure(image="", text="Open an image to begin" if key == "input"
                             else "No prediction yet")
            self.photos.pop(key, None)
            return
        thumbnail = image.copy()
        thumbnail.thumbnail((max(20, width or widget.winfo_width() - 10),
                             max(20, height or widget.winfo_height() - 10)))
        photo = ImageTk.PhotoImage(thumbnail, master=self.root)
        self.photos[key] = photo
        widget.configure(image=photo, text="")

    def open_image(self):
        path = filedialog.askopenfilename(title="Choose a local image",
                                          filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff")])
        if not path:
            return
        try:
            image = load_image(path)
        except Exception as exc:
            messagebox.showerror("Could not open image", str(exc))
            return
        self.image, self.path, self.result = image, Path(path), None
        self.preview("input")
        self.preview("output")
        self.segment.configure(state="normal")
        self.export.configure(state="disabled")
        self.status.configure(text=f"{self.path.name} · Enter target names, separated by commas.")

    def infer(self, image, text):
        if self.segmenter is None:
            self.segmenter = Segmenter()
        return self.segmenter.run(image, text)

    def run(self):
        if self.busy or self.image is None:
            return
        text = self.prompt.get()
        try:
            parse_prompts(text)
        except ValueError as exc:
            self.status.configure(text=str(exc))
            return
        self.busy = True
        self.result = None
        self.preview("output")
        for widget in (self.pick, self.entry, self.segment, self.export):
            widget.configure(state="disabled")
        self.status.configure(text="Running on CPU. First run may fetch model files and take longer…")
        future = self.pool.submit(self.infer, self.image, text)
        self.root.after(100, self.poll, future)

    def poll(self, future):
        if not future.done():
            self.root.after(100, self.poll, future)
            return
        self.busy = False
        for widget in (self.pick, self.entry, self.segment):
            widget.configure(state="normal")
        try:
            self.result = future.result()
        except Exception as exc:
            self.status.configure(text=f"Segmentation failed: {exc}")
            return
        _, instances, meta = self.result
        self.preview("output")
        self.export.configure(state="normal")
        outcome = f"{len(instances)} mask(s)" if instances else "No masks found"
        self.status.configure(text=f"{outcome} · {', '.join(meta['prompts'])} · "
                                   f"{meta['predict_call_ms']:.0f} ms prediction call")

    def save(self):
        if self.result is None:
            return
        parent = filedialog.askdirectory(title="Choose a parent folder for the export")
        if not parent:
            return
        directory = Path(parent) / strftime("yoloe-%Y%m%d-%H%M%S")
        try:
            save_result(directory, *self.result)
            self.status.configure(text=f"Saved to {directory}")
        except OSError as exc:
            messagebox.showerror("Could not export masks", str(exc))

    def close(self):
        if self.busy:
            messagebox.showinfo("YOLOE-26 is running", "Wait for the current run to finish before closing.")
            return
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    PromptLab(root)
    root.mainloop()

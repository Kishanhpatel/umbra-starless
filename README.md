# Umbra Starless (BETA)

Umbra Starless is an AI star remover for [Siril](https://siril.org). It works on linear or stretched images. It is a Siril Python script with an AI model. It runs locally on your computer and is free. It removes stars and their halos and fills the holes, and it can give you the star layer on its own.

**Status: Beta (v0.2.1).** Expect rough edges.

![Umbra Starless panel](docs/screenshots/Umbra_Starless_panel.png)


![Umbra Starless panel](docs/screenshots/Umbra_Starless_panel_updnote.png)

## Download and install

1. Download the single file [`Umbra_Starless.py`](https://raw.githubusercontent.com/Kishanhpatel/umbra-starless/main/Umbra_Starless.py) (also attached to the [latest release](https://github.com/Kishanhpatel/umbra-starless/releases/latest)). Save it in any folder you like. Siril has no standard scripts folder name.
2. In Siril open **Preferences > Scripts**, add that folder, refresh the list, then run **Umbra Starless** from the Scripts menu.
3. First run: Siril installs the Python packages the script needs (numpy, onnxruntime, PyQt6, scipy, astropy). This can take a few minutes.
4. If no model is installed, press **Download model** (Umbra Starless v1.0, about 100 MB). The file is checked with SHA-256 and saved in a folder named `Umbra Starless` next to the script.

**Auto-keep model.** Auto-keep (see below) needs its own small model (about 3 MB). **Download model** and **Check for updates** offer it next to the star remover models. It is only needed when **Auto-keep** is ticked; without it the box says "Auto-keep needs the Auto-keep model. Press Download model." and stays unticked.

**Manual install.** Download the `.onnx` files from the [release page](https://github.com/Kishanhpatel/umbra-starless/releases/latest) and put them in the `Umbra Starless` folder next to the script. To use another folder, set `UMBRA_STARLESS_MODEL_DIR` to it. `UMBRA_STARLESS_MODEL` can point at one specific model file.

**Updates.** **Check for updates** compares your installed models and the script version with the published list (`models/models.json`). A new or changed model downloads only when you press the button. A newer script is shown as a link to the release page; the script never replaces itself. Nothing is sent about you or your computer (only a generic client name). With no network the installed models keep working. Set `UMBRA_STARLESS_OFFLINE=1` (or `UMBRA_OFFLINE=1`) to turn the lookup off.

## What it does

The model runs once over the whole image in 512 px tiles. Controls:

- **Star removal** (0 to 100%, default 100): 100 removes all stars. Lower values make every star smaller and dimmer; faint stars go first. 0 leaves the original image.
- **Auto-keep nebula found in the star layer** (off by default): a small checker network looks at the star layer and puts nebula filaments, dust-lane edges or galaxy detail that were removed by mistake back into the starless image, like a Keep mark. Star cores are never put back. It is optional and results vary by image: on many images it changes little, so judge it by eye. Paint Keep marks for anything it misses.
- **Data** (Auto, Linear, Stretched): Auto guesses from the image. Override it if the guess is wrong.
- **Output** (Starless image, Stars only): Stars only gives the full star layer and ignores Star removal.
- **Also save the star layer as FITS next to the image**: saves `<image>_stars.fit` in the same folder (a new file; the image is not touched).

The star layer is the input minus the starless result for linear data, and recombines with it by screen blending for stretched data. Starless plus stars gives back the input (to float rounding).


**Preview and correction marks.** **Preview** removes the stars from the whole image once, at full resolution, and opens an editor window (the network is the slow part; everything after it is quick). The window zooms with the mouse wheel and pans by dragging (**Navigate**); the image is shown auto-stretched when it is linear, and only the display is downsampled, never the data. Hold **Show original** (or the Space key) to compare and release to see the result again; **Show star layer** shows the star layer. Moving **Star removal** in the main window updates the result at once from the stored result. Nothing is written to Siril from the editor.

**Correction marks** (brush size 1 to 100, default 30):
- **Remove** (red): paint where stars or residue were wrongly kept. **Apply marks** (or **Apply Remove marks after each stroke**) re-runs the network only on the tiles the marks touch, then blends the result in.
- **Keep** (green): paint where something was wrongly removed (a galaxy or nebula knot). The original comes back there at once.
- **Eraser** (or right mouse button) erases marks; **Undo stroke** (Ctrl+Z) and **Clear marks** undo them; **Show marks** hides the overlay. Marks are drawn on the picture as Siril shows it and are mapped to the image rows correctly, including Siril's flipped row order.

The editor keeps its result: **Apply** in the main window uses the stored result and the marks when the image and model are the same (so it does not run the network again), otherwise it computes from scratch. Marks belong to the image that was previewed; if the image in Siril changed, they are not used. The editor holds the full-size result in memory (several copies of the image), so very large frames need a few GB.

**Apply.** **Apply** replaces the image in Siril with the result. Use Siril's **Undo** to go back. Siril's scripting interface cannot write a FITS HISTORY line, so the settings used are written to the Siril log (and into any extra FITS file the tool saves).

**Command line.** `python Umbra_Starless.py --image IN.fits --out OUT.fits` runs the tool without a window. Flags: `--removal 0..1`, `--data auto|linear|stretched`, `--output starless|stars`, `--stars-out FILE`, `--auto-keep` (with `--critic-model PATH`, default: the installed one), `--remove-marks MAP.fits` and `--keep-marks MAP.fits` (maps from 0 to 1, same size as the image, rows as stored in the file), `--model PATH`. `--check-updates` and `--download [MODEL_ID|all]` work from the command line too, and `--help` lists everything.

## Models

Since v0.2.0 the model files store their weights at half size (16-bit) and compute in full 32-bit precision, so the results are bit-identical to the larger v0.1.x files. If you already have the larger file it keeps working and is not downloaded again.

Models are published as GitHub release assets and are not stored in this repository. See `models/models.json`.

| Name | File | Size | Notes |
|---|---|---|---|
| Umbra Starless v1.0 (default) | `umbra_starless_v1.0.onnx` | 100 MB | The main model. |
| Umbra Starless v1.1 (experimental) | `umbra_starless_v1.1_experimental.onnx` | 100 MB | A small gain over v1.0 in our tests. Judge by eye on your image. |
| Umbra Starless Auto-keep model v1.0 | `umbra_starless_critic_v1.0.onnx` | 3 MB | Only for the Auto-keep option. Not a star remover; it is not listed in the model menu. |

## Strengths

- Removes stars and halos on both linear and stretched images, and the star layer is available as its own output.
- The Star removal slider reduces stars gradually instead of all or nothing, and updates at once in the editor.
- Correction brushes (Remove and Keep) for the places the model gets wrong, with an optional Auto-keep for nebula and galaxy detail that ended up in the star layer.
- Starless plus star layer gives back the input image.
- Fills the holes from the surroundings. Noise in the filled areas matches the rest of the image.
- Handles mono images and images with NaN holes (stacking borders); NaN pixels stay NaN.
- GPU acceleration with automatic CPU fallback.

## Weaknesses

- Very bright stars (for example Alnitak, or the stars in NGC 2264 and the Wizard Nebula) can leave glows or starburst artefacts.
- Bright nebula and galaxy cores can be partly taken for stars. The core of the Orion Nebula (M42) is still partly cut and leaves a messy patch.
- Compact knots in galaxies (M82, spiral-arm knots in M33) and thin filaments (Veil, Crescent) can partly go into the star layer.
- Auto-keep is conservative and optional: on many images it puts back little or nothing, and it can miss structure. Paint Keep marks where it does.
- Remove marks are a hint to the network, not a guarantee: a mark may need a second stroke or a smaller or larger brush.
- Images smaller than 512 px on a side are processed differently from the original near the bottom edge (see Testing).
- Big frames are heavy: a 6016 x 6016 colour frame took about 70 s on an Apple M3 (Core ML) and about 7.5 GB of memory; the editor keeps extra copies of the image, so very large frames need more.
- Tested mainly on the author's own data and one Mac.

## GPU acceleration

The GPU is used by default whenever one can be used, and the CPU only as a fallback. Providers are tried in this order: CUDA (NVIDIA), DirectML (Windows), ROCm and MIGraphX (AMD on Linux), OpenVINO (Intel), Core ML (Mac), then CPU. Each one is checked against the CPU result on a small probe tile before it is used; if it fails or the result differs, the script falls back to the CPU. The status line says which one ran ("Ran on ..."). Set `UMBRA_STARLESS_CPU=1` to force the CPU.

On first run the script installs the onnxruntime package that gives your system a GPU:

- **Mac:** `onnxruntime` (Core ML, Apple GPU, built in).
- **Windows:** `onnxruntime-directml` (any DirectX 12 GPU: NVIDIA, AMD or Intel; no extra downloads).
- **Linux with an NVIDIA GPU:** `onnxruntime-gpu` (CUDA). If the system has no CUDA 12 / cuDNN 9 libraries, NVIDIA's own pip packages of them are installed too (a one-time download of about 2 to 3 GB).
- **Linux with AMD or Intel graphics:** there is no GPU build that installs automatically, so these run on the CPU. If you install `onnxruntime-rocm`, `onnxruntime-migraphx` or `onnxruntime-openvino` yourself in Siril's Python, the script uses it.

## Testing

Be aware of what has and has not been tried. This is a beta.

Tested:
- macOS 26 on an Apple M3, with Core ML and with the CPU.
- Standalone (command line and window test) and inside Siril 1.4.5 run headless (`siril-cli` with `pyscript`): read the image, process it, write it back with an Undo state, save.
- The windows (panel and editor) were tested offscreen only, with synthetic mouse strokes.

NOT tested:
- Windows and Linux.
- CUDA, DirectML and ROCm (the code is written; please report what you see).
- Intel Macs.
- The interactive windows inside the Siril GUI (the editor included), and painting with a real mouse or pen.
- Very large frames in the editor (memory use is several times the image size).

Correction marks and Auto-keep were compared with the original PyTorch version on three 1024 px crops with synthetic marks: Remove and Keep marks, Auto-keep and their combination differ by at most about 2 noise sigma at a single pixel (the same threshold outlier as without marks) and by about 0.02 sigma rms or less; the Auto-keep map differs by 2e-7. The Auto-keep model flagged only 0.05 to 0.2 percent of those crops. Remove marks use a local noise fill like the original script.

Parity with the original PyTorch version: the whole pipeline (default settings, three removal amounts, star layer) differs by at most about 0.3 noise sigma at a single pixel (3 sigma at one pixel of one image at partial removal) and about 1e-4 sigma rms, against changes of tens to hundreds of sigma. Cases: five real crops (up to 2048 px), mono, a NaN hole, stretched data. Core ML and CPU agree.
Small images: the model uses fixed 512 px tiles, so images under 512 px are padded by mirroring. A 300 x 400 crop differed near the padded bottom edge (rms 5 sigma overall, 0.2 to 0.7 sigma in the interior, against a change of 47 sigma). This cannot be matched with a fixed 512 px model. Recombining starless and stars is still exact.

## Layout

- `Umbra_Starless.py`: the script (v0.2.1, ONNX Runtime, with model download and update check)
- `models/models.json`: model manifest (name, version, download URL, SHA-256, size)
- `docs/screenshots/`: panel and editor screenshots
- `LICENSE`, `LICENSE-MODELS.md`: licenses

## Contact

Bugs or questions: umbraastrolabs@gmail.com, or open an issue on GitHub.

## License

- Script: GPL-3.0-only (see `LICENSE`).
- Models: CC BY-NC-SA 4.0, non-commercial (see `LICENSE-MODELS.md`).

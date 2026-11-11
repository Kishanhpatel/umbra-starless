# Umbra Starless (BETA)

Umbra Starless is an AI star remover for [Siril](https://siril.org). It works on linear or stretched images. It is a Siril Python script with an AI model. It runs locally on your computer and is free. It removes stars and their halos and fills the holes, and it can give you the star layer on its own.

**Status: Beta (v0.1.0).** Expect rough edges.

![Umbra Starless panel](docs/screenshots/Umbra_Starless_panel.png)

![Umbra Starless panel](docs/screenshots/Umbra_Starless_panel_updnote.png)

## Download and install

1. Download the single file [`Umbra_Starless.py`](https://raw.githubusercontent.com/Kishanhpatel/umbra-starless/main/Umbra_Starless.py) (also attached to the [latest release](https://github.com/Kishanhpatel/umbra-starless/releases/latest)). Save it in any folder you like. Siril has no standard scripts folder name.
2. In Siril open **Preferences > Scripts**, add that folder, refresh the list, then run **Umbra Starless** from the Scripts menu.
3. First run: Siril installs the Python packages the script needs (numpy, onnxruntime, PyQt6, scipy, astropy). This can take a few minutes.
4. If no model is installed, press **Download model** (Umbra Starless v1.0, about 199 MB). The file is checked with SHA-256 and saved in a folder named `Umbra Starless` next to the script.

**Manual install.** Download the `.onnx` file from the [release page](https://github.com/Kishanhpatel/umbra-starless/releases/latest) and put it in the `Umbra Starless` folder next to the script. To use another folder, set `UMBRA_STARLESS_MODEL_DIR` to it. `UMBRA_STARLESS_MODEL` can point at one specific model file.

**Updates.** **Check for updates** compares your installed models and the script version with the published list (`models/models.json`). A new or changed model downloads only when you press the button. A newer script is shown as a link to the release page; the script never replaces itself. Nothing is sent about you or your computer (only a generic client name). With no network the installed models keep working. Set `UMBRA_STARLESS_OFFLINE=1` (or `UMBRA_OFFLINE=1`) to turn the lookup off.

## What it does

The model runs once over the whole image in 512 px tiles. Controls:

- **Star removal** (0 to 100%, default 100): 100 removes all stars. Lower values make every star smaller and dimmer; faint stars go first. 0 leaves the original image.
- **Data** (Auto, Linear, Stretched): Auto guesses from the image. Override it if the guess is wrong.
- **Output** (Starless image, Stars only): Stars only gives the full star layer and ignores Star removal.
- **Also save the star layer as FITS next to the image**: saves `<image>_stars.fit` in the same folder (a new file; the image is not touched).

The star layer is the input minus the starless result for linear data, and recombines with it by screen blending for stretched data. Starless plus stars gives back the input (to float rounding).


**Preview.** The **Preview** button opens a separate, resizable window. It shows a region of the image before and after (same stretch for both): your current Siril selection if it is at least 64 px, otherwise a centred 768 x 768 crop (the whole image if smaller). Small selections are processed with some surrounding image for context, then cropped back. Hold down **Show original** (or the Space key) to see the original and release it to see the result again; **Show star layer** shows the star layer. It never changes the image in Siril.

**Apply.** **Apply** replaces the image in Siril with the result. Use Siril's **Undo** to go back. Siril's scripting interface cannot write a FITS HISTORY line, so the settings used are written to the Siril log (and into any extra FITS file the tool saves).

**Command line.** `python Umbra_Starless.py --image IN.fits --out OUT.fits` runs the tool without a window. Flags: `--removal 0..1`, `--data auto|linear|stretched`, `--output starless|stars`, `--stars-out FILE`, `--model PATH`. `--check-updates` and `--download [MODEL_ID|all]` work from the command line too, and `--help` lists everything.

## Models

Models are published as GitHub release assets and are not stored in this repository. See `models/models.json`.

| Name | File | Size | Notes |
|---|---|---|---|
| Umbra Starless v1.0 (default) | `umbra_starless_v1.0.onnx` | 199 MB | The main model. |
| Umbra Starless v1.1 (experimental) | `umbra_starless_v1.1_experimental.onnx` | 199 MB | A small gain over v1.0 in our tests. Judge by eye on your image. |

## Strengths

- Removes stars and halos on both linear and stretched images, and the star layer is available as its own output.
- The Star removal slider reduces stars gradually instead of all or nothing.
- Starless plus star layer gives back the input image.
- Fills the holes from the surroundings. Noise in the filled areas matches the rest of the image.
- Handles mono images and images with NaN holes (stacking borders); NaN pixels stay NaN.
- GPU acceleration with automatic CPU fallback.

## Weaknesses

- Very bright stars (for example Alnitak, or the stars in NGC 2264 and the Wizard Nebula) can leave glows or starburst artefacts.
- Bright nebula and galaxy cores can be partly taken for stars. The core of the Orion Nebula (M42) is still partly cut and leaves a messy patch.
- Compact knots in galaxies (M82, spiral-arm knots in M33) and thin filaments (Veil, Crescent) can partly go into the star layer.
- There are no correction brushes in this version. Workarounds: lower Star removal, or save the star layer and blend the two layers yourself.
- Images smaller than 512 px on a side are processed differently from the original near the bottom edge (see Testing).
- Tested mainly on the author's own data and one Mac.

## GPU acceleration

Providers are tried in this order: CUDA, DirectML (Windows), ROCm (AMD on Linux), Core ML (Mac), then CPU. Each one is checked against the CPU result on a small probe tile before it is used. If the result differs, or the provider fails, the script falls back to the CPU. The status line says which one ran ("Ran on ..."). Set `UMBRA_STARLESS_CPU=1` to force the CPU. On first run the script installs the matching onnxruntime package for your system: `onnxruntime-directml` (Windows), `onnxruntime-gpu` (Linux with an NVIDIA GPU) or plain `onnxruntime` (everything else).

## Testing

Be aware of what has and has not been tried. This is a beta.

Tested:
- macOS 26 on an Apple M3, with Core ML and with the CPU.
- Standalone (command line and window test) and inside Siril 1.4.5 run headless (`siril-cli` with `pyscript`): read the image, process it, write it back with an Undo state, save.
- The window was tested offscreen only.

NOT tested:
- Windows and Linux.
- CUDA, DirectML and ROCm (the code is written; please report what you see).
- Intel Macs.
- The interactive window inside the Siril GUI.

Parity with the original PyTorch version: the whole pipeline (default settings, three removal amounts, star layer) differs by at most about 0.3 noise sigma at a single pixel (3 sigma at one pixel of one image at partial removal) and about 1e-4 sigma rms, against changes of tens to hundreds of sigma. Cases: five real crops (up to 2048 px), mono, a NaN hole, stretched data. Core ML and CPU agree.
Small images: the model uses fixed 512 px tiles, so images under 512 px are padded by mirroring. A 300 x 400 crop differed near the padded bottom edge (rms 5 sigma overall, 0.2 to 0.7 sigma in the interior, against a change of 47 sigma). This cannot be matched with a fixed 512 px model. Recombining starless and stars is still exact.

## Layout

- `Umbra_Starless.py`: the script (v0.1.0, ONNX Runtime, with model download and update check)
- `models/models.json`: model manifest (name, version, download URL, SHA-256, size)
- `docs/screenshots/`: panel screenshots
- `LICENSE`, `LICENSE-MODELS.md`: licenses

## Contact

Bugs or questions: umbraastrolabs@gmail.com, or open an issue on GitHub.

## License

- Script: GPL-3.0-only (see `LICENSE`).
- Models: CC BY-NC-SA 4.0, non-commercial (see `LICENSE-MODELS.md`).

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Umbra Starless (BETA), Copyright (c) 2026 Kishan Patel. Own code; no code taken from other scripts.
"""AI star remover for linear or stretched images. Preview or Apply; Apply replaces the image in Siril (Undo available).
Star removal 100% removes the stars, lower values only reduce them. Output can be the starless image or the star layer.
Run in Siril (Scripts menu), or standalone:
  python Umbra_Starless.py --image in.fits --out out.fits [--model PATH.onnx] [--removal 0..1] [--data auto|linear|stretched] [--output starless|stars] [--stars-out stars.fits]
  python Umbra_Starless.py --ui-test        (opens the window on a fake image, runs Preview once, exits)
  python Umbra_Starless.py --check-updates | --download [MODEL_ID|all]"""
VERSION = '0.1.0'
# First family release (ONNX Runtime, no PyTorch): one-button Preview / Apply flow (the correction marks and Auto-keep of the old script are not in this release).
# ============================================================================================ shared code (src/common.py)
# Everything the five scripts share: dependency install, ONNX Runtime sessions, updater, preferences, Siril I/O, family UI, CLI.
import sys, os, json, hashlib, argparse, time, math, re, types
from pathlib import Path
_STANDALONE = any(a in ('--image', '--ui-test', '--download', '--check-updates') or a.startswith(('--image=', '--download=')) for a in sys.argv)
if __name__ == '__main__' and not _STANDALONE and '--help' not in sys.argv and '-h' not in sys.argv:
    import sirilpy as _siril_deps
    def _ort_requirement():
        """onnxruntime variant for this machine (all variants import as 'onnxruntime'; only one may be installed). An existing one is kept."""
        import importlib.metadata as _md, platform as _pf, shutil as _sh
        for n in ('onnxruntime-gpu', 'onnxruntime-directml', 'onnxruntime-rocm', 'onnxruntime'):
            try:
                _md.version(n); return n
            except Exception:
                pass
        if _pf.system() == 'Windows': return 'onnxruntime-directml'          # GPU through DirectX 12 (NVIDIA / AMD / Intel); CUDA needs onnxruntime-gpu
        if _pf.system() == 'Linux' and _sh.which('nvidia-smi'): return 'onnxruntime-gpu'   # needs the CUDA 12 + cuDNN 9 libraries; falls back to CPU without them
        return 'onnxruntime'                                                   # macOS: Core ML is built in; Linux without NVIDIA: CPU
    _siril_deps.ensure_installed('numpy', _ort_requirement(), 'PyQt6', 'scipy', 'astropy')
import numpy as np
import urllib.request

SCRIPT_DIR = Path(__file__).resolve().parent
ENGINES = {}      # tool key -> engine namespace (filled by the build)
TOOLS = {}        # tool key -> ToolSpec (filled by the build)
USER_AGENT = 'UmbraAstro'    # generic client name only; nothing about the user or the machine is sent
_REDIRECT_HOSTS = {'github.com', 'objects.githubusercontent.com', 'release-assets.githubusercontent.com', 'raw.githubusercontent.com'}
GITHUB_USER = 'Kishanhpatel'


class ToolSpec:
    """Everything that differs between the tools (the shared code reads only this)."""
    def __init__(self, **kw):
        self.legacy_repos = ()
        self.fallback_folder = None
        self.labels = {}
        self.default_stems = ()
        self.notes = []
        self.linear_only = True
        self.preview_whole = False       # Gradient: preview runs on the whole image
        self.reuse_preview = False       # Gradient: Apply with unchanged settings reuses the preview result
        self.alt = None                  # (button text, info key) second preview layer
        self.progress_unit = 'tile'
        self.__dict__.update(kw)

    @property
    def title(self):
        return f'{self.name} (BETA)'

    # hooks (overridden by the tool modules)
    def build_panel(self, ui, layout, host):
        raise NotImplementedError
    def add_cli(self, ap):
        pass
    def params_from_args(self, a):
        return {}
    def image_params(self, header_text):
        return {}
    def preview_params(self, params):
        return params
    def label(self, params):
        return ''
    def output(self, params, out, info):
        return out
    def post_apply(self, ctx, params, info):
        return ''
    def check_cli(self, a, ap):
        pass
    def post_cli(self, a, dst, info, header, label):
        return ''


def env(spec, name, default=None):
    return os.environ.get(spec.prefix + name, default)


def offline_mode(spec):
    return bool(env(spec, 'OFFLINE') or os.environ.get('UMBRA_OFFLINE'))


# ----------------------------------------------------------------------------------------------- models folder
def _onnx_files(d):
    try:
        return sorted(Path(d).glob('*.onnx')) if Path(d).is_dir() else []
    except OSError:
        return []


def model_dir(spec):
    """(folder, used_fallback). <PREFIX>MODEL_DIR wins; else '<folder>' next to the script; Deblur falls back to the old
    'Umbra Deconvolve' folder when its own is missing or empty and the old one has models."""
    e = env(spec, 'MODEL_DIR')
    if e:
        return Path(e), False
    main = SCRIPT_DIR / spec.folder
    if spec.fallback_folder and not _onnx_files(main) and _onnx_files(SCRIPT_DIR / spec.fallback_folder):
        return SCRIPT_DIR / spec.fallback_folder, True
    return main, False


def download_dir(spec):
    e = env(spec, 'MODEL_DIR')
    return Path(e) if e else SCRIPT_DIR / spec.folder


_MANIFEST_NAMES = {}      # (tool, file stem) -> name from the manifest (after an update check)


def model_label(spec, stem):
    return spec.labels.get(stem) or _MANIFEST_NAMES.get((spec.key, stem)) or stem


def list_models(spec):
    items = [(model_label(spec, f.stem), str(f)) for f in _onnx_files(model_dir(spec)[0])]
    items.sort(key=lambda it: (0 if Path(it[1]).stem in spec.default_stems else 1, it[0]))
    return items


def default_model_path(spec, saved=None):
    e = env(spec, 'MODEL')
    if e:
        return e
    d = model_dir(spec)[0]
    if saved:
        p = d / Path(saved).name
        if p.suffix == '.onnx' and p.is_file():
            return str(p)
    items = list_models(spec)
    return items[0][1] if items else None


# ----------------------------------------------------------------------------------------------- preferences
class Prefs:
    """Umbra_<Tool>_prefs.json: {"model": <file name>, "settings": {...}}; combined: {"tool": key, "tools": {key: {...}}}."""
    def __init__(self, path, combined=False, enabled=True):
        self.path, self.combined, self.enabled = Path(path), combined, enabled

    def _read(self):
        try:
            d = json.loads(self.path.read_text())
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def load(self, key):
        d = self._read()
        if self.combined:
            d = (d.get('tools') or {}).get(key) or {}
        s = d.get('settings')
        return (d.get('model') if isinstance(d.get('model'), str) else None), (s if isinstance(s, dict) else {})

    def load_tool(self):
        t = self._read().get('tool')
        return t if isinstance(t, str) else None

    def save(self, key, model, settings, current=None):
        if not self.enabled:
            return
        try:
            entry = {'model': Path(model).name if model else None, 'settings': settings}
            if self.combined:
                d = self._read()
                tools = d.get('tools') if isinstance(d.get('tools'), dict) else {}
                tools[key] = entry
                out = {'tool': current or key, 'tools': tools}
            else:
                out = entry
            self.path.write_text(json.dumps(out, indent=1))
        except Exception:
            pass


# ----------------------------------------------------------------------------------------------- ONNX Runtime
_IMPORT_ERROR = {}   # first onnxruntime / scipy import failure (remembered so every later attempt shows the real cause)


def log_file(spec):
    return SCRIPT_DIR / f'{spec.script}_error.log'


def write_error_log(spec, text):
    try:
        log_file(spec).write_text(f'{time.ctime()}\npython {sys.executable}\n\n{text}')
    except Exception:
        pass


def import_engine(spec):
    """Import onnxruntime and scipy exactly once; on failure keep the ORIGINAL error (with traceback) for every later attempt."""
    if _IMPORT_ERROR:
        raise RuntimeError(_IMPORT_ERROR['msg'])
    try:   # Siril's working folder may have been deleted; some imports then fail in os.getcwd() -> move to a folder that exists
        os.getcwd()
    except OSError:
        os.chdir(str(SCRIPT_DIR))
    try:
        import onnxruntime  # noqa: F401
        import scipy.ndimage  # noqa: F401
    except BaseException as e:
        import traceback
        tb = traceback.format_exc()
        _IMPORT_ERROR['msg'] = f'{type(e).__name__}: {e} (full details: {log_file(spec)})'
        write_error_log(spec, f'sys.path {sys.path}\n\n{tb}')
        print(tb, flush=True)
        raise RuntimeError(_IMPORT_ERROR['msg']) from e


def make_session_factory(prefix):
    """-> factory(model_path, feed) -> (run_fn, provider_label, note).  run_fn(feed_dict) -> list of outputs (like sess.run(None, feed)).
    GPU providers are tried in order of preference for the platform (CUDA, DirectML, ROCm, Core ML); each is checked against the CPU
    provider on one probe feed (outputs must agree to 1e-3 relative) and the first that works is used. Anything missing, failing or
    disagreeing falls through to the next, and finally to the CPU.  <PREFIX>CPU=1 forces the CPU."""
    def factory(model_path, feed):
        import onnxruntime as ort
        path = str(model_path)
        so = ort.SessionOptions(); so.log_severity_level = 3
        holder = []
        def mk_cpu():
            if not holder:
                holder.append(ort.InferenceSession(path, so, providers=['CPUExecutionProvider']))
            return holder[0]
        note = None
        sess, label = None, 'CPU'
        if not os.environ.get(prefix + 'CPU'):
            avail = ort.get_available_providers()
            cands = [('CUDAExecutionProvider', 'CUDA', {'use_tf32': '0', 'cudnn_conv_algo_search': 'HEURISTIC'}, False),
                     ('DmlExecutionProvider', 'DirectML', {}, True),
                     ('ROCMExecutionProvider', 'ROCm', {}, False),
                     ('CoreMLExecutionProvider', 'Core ML', {'ModelFormat': 'MLProgram', 'MLComputeUnits': 'CPUAndGPU'}, False)]
            notes = []
            for name, lab, opts, dml in cands:
                if name not in avail:
                    continue
                try:
                    so2 = ort.SessionOptions(); so2.log_severity_level = 3
                    if dml:                                    # DirectML requirements
                        so2.enable_mem_pattern = False; so2.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    g = ort.InferenceSession(path, so2, providers=[(name, opts), 'CPUExecutionProvider'])
                    if name not in g.get_providers():
                        raise RuntimeError('provider did not load')
                    a, b = g.run(None, feed), mk_cpu().run(None, feed)
                    err = max(float(np.abs(x - y).max() / max(np.abs(y).max(), 1e-9)) for x, y in zip(a, b))
                    if np.isfinite(err) and err < 1e-3:
                        sess, label = g, lab
                        break
                    notes.append(f'{lab} result differed from the CPU ({err:.1e})')
                except Exception as e:
                    notes.append(f'{lab} unavailable ({str(e).strip().splitlines()[0][:90] if str(e).strip() else type(e).__name__})')
            if sess is None and notes:
                note = '; '.join(notes) + '; using the CPU.'
            elif sess is None:
                note = 'No GPU provider in this onnxruntime build; using the CPU.'
        if sess is None:
            sess = mk_cpu()
        return (lambda f: sess.run(None, f)), label, note
    return factory


_ENGINE_CACHE = {}     # tool key -> (model path, Engine)   (one model in memory per tool; the combined window unloads the others)


def get_engine(spec, model_path):
    c = _ENGINE_CACHE.get(spec.key)
    if c and c[0] == str(model_path):
        return c[1]
    _ENGINE_CACHE.pop(spec.key, None)
    import_engine(spec)
    p = Path(str(model_path or ''))
    if not p.is_file():
        raise FileNotFoundError(f'Model not found: {p}')
    eng = ENGINES[spec.key].Engine(str(p), make_session_factory(spec.prefix))
    _ENGINE_CACHE[spec.key] = (str(p), eng)
    return eng


def unload_engines(keep=None):
    for k in list(_ENGINE_CACHE):
        if k != keep:
            _ENGINE_CACHE.pop(k, None)
    import gc; gc.collect()


def stretched_warning(data):
    """Message if the data looks stretched (median > 0.2 of max), else None."""
    d = np.asarray(data, np.float32)
    fin = d[np.isfinite(d)]
    if fin.size == 0:
        return None
    mx = float(fin.max())
    if mx > 0 and float(np.median(fin)) > 0.2 * mx:
        return 'the data looks stretched (median above 20% of the maximum); this tool expects linear data. Running anyway.'
    return None


# ----------------------------------------------------------------------------------------------- model download / updates
class DownloadError(Exception):
    pass


class UpdSpec:
    """Where a script looks for models: its manifest repo and every repo whose release assets / release page it accepts."""
    def __init__(self, repo, allowed=None):
        self.repo = repo
        self.allowed = list(allowed or [repo])
        self.manifest_url = f'https://raw.githubusercontent.com/{repo}/main/models/models.json'
        self.releases_url = f'https://github.com/{repo}/releases'


def updspec_for(spec):
    return UpdSpec(spec.repo, [spec.repo, *spec.legacy_repos])


def combined_upd(specs):
    """The combined script's own manifest repo; it accepts model / script URLs from the repos of all its tools (no re-upload)."""
    me = f'{GITHUB_USER}/umbra-astro-ai'
    return UpdSpec(me, [me] + [t.repo for t in specs] + [r for t in specs for r in t.legacy_repos])


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        from urllib.parse import urlparse
        u = urlparse(newurl)
        if u.scheme != 'https' or u.hostname not in _REDIRECT_HOSTS:
            raise DownloadError(f'Refused a redirect to {u.hostname or newurl}')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _ssl_context():
    """Default certificate store; if the Python build has none (python.org macOS builds without 'Install Certificates'), use certifi's.
    Certificates are always verified."""
    import ssl
    ctx = ssl.create_default_context()
    if not ctx.cert_store_stats().get('x509_ca') and not ctx.cert_store_stats().get('x509'):
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            pass
    return ctx


def _open(url, timeout=20):
    """The only network entry point: HTTPS GET with a generic user agent; redirects only to GitHub hosts; certificates verified."""
    opener = urllib.request.build_opener(_SafeRedirect(), urllib.request.HTTPSHandler(context=_ssl_context()))
    return opener.open(urllib.request.Request(url, headers={'User-Agent': USER_AGENT}), timeout=timeout)


def _valid_model_url(url, file, upd):
    from urllib.parse import urlparse
    u = urlparse(str(url))
    return (u.scheme == 'https' and u.hostname == 'github.com' and not u.query and u.path.endswith('/' + file)
            and any(u.path.startswith(f'/{r}/releases/download/') for r in upd.allowed))


def _valid_entry(m, upd):
    try:
        f = m['file']
        return (re.fullmatch(r'[A-Za-z0-9._-]{1,120}\.onnx', f) is not None and '..' not in f and _valid_model_url(m['url'], f, upd)
                and re.fullmatch(r'[0-9a-f]{64}', str(m['sha256']).lower()) is not None and int(m['size_bytes']) > 0 and bool(m.get('name')))
    except Exception:
        return False


def _version_tuple(v):
    try:
        return tuple(int(p) for p in str(v).strip().lstrip('v').split('.'))
    except Exception:
        return None


def parse_manifest(data, upd):
    """Validated manifest dict {'models': [...valid entries...], 'script': {'version','url'} or None}. Bad entries and entries marked
    "legacy" (kept only for old script versions) are dropped."""
    if not isinstance(data, dict) or data.get('schema') != 1 or not isinstance(data.get('models'), list):
        raise DownloadError('The model list has an unexpected format.')
    models = [dict(m, sha256=str(m['sha256']).lower(), size_bytes=int(m['size_bytes'])) for m in data['models']
              if isinstance(m, dict) and not m.get('legacy') and _valid_entry(m, upd)]
    sc = data.get('script')
    if not (isinstance(sc, dict) and _version_tuple(sc.get('version')) and any(str(sc.get('url', '')).startswith(f'https://github.com/{r}/') for r in upd.allowed)):
        sc = None
    return dict(models=models, script=sc)


_MANIFEST_CACHE = {}


def fetch_manifest(upd, url=None, use_cache=False):
    key = url or upd.manifest_url
    c = _MANIFEST_CACHE.get(key)
    if use_cache and c and time.time() - c[0] < 300:
        return c[1]
    try:
        with _open(key, 20) as resp:
            raw = resp.read(1_000_000)
        man = parse_manifest(json.loads(raw.decode('utf-8')), upd)
        _MANIFEST_CACHE[key] = (time.time(), man)
        return man
    except DownloadError:
        raise
    except Exception as e:
        why = ' Certificate problem: on macOS run "Install Certificates.command" for your Python.' if 'CERTIFICATE' in str(e).upper() else ''
        raise DownloadError(f'Could not reach the model list ({type(e).__name__}). Offline? The installed models still work.{why}') from e


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def model_state(entry, folder):
    """'missing' | 'current' | 'update' for a manifest entry against the installed file."""
    p = Path(folder) / entry['file']
    if not p.is_file():
        return 'missing'
    return 'current' if file_sha256(p) == entry['sha256'] else 'update'


def entries_for(manifest, tool=None):
    """Manifest entries for one tool (entries with a 'tool' field belong to that tool; without one they belong to every single script)."""
    return [m for m in manifest['models'] if tool is None or m.get('tool') in (None, tool)]


def check_updates(manifest, spec, folder=None):
    """-> dict(models=[(entry, state)], script_version, script_url, script_newer)."""
    folder = folder or download_dir(spec)
    sc = manifest.get('script')
    newer = bool(sc) and _version_tuple(sc['version']) > (_version_tuple(VERSION) or (0,))
    for m in manifest['models']:
        _MANIFEST_NAMES[(m.get('tool') or spec.key, Path(m['file']).stem)] = m['name']
    return dict(models=[(m, model_state(m, folder)) for m in entries_for(manifest, spec.key)],
                script_version=sc['version'] if sc else None, script_url=sc['url'] if sc else None, script_newer=newer)


def download_model(entry, upd, folder, progress=None, cancel=None):
    """Download one manifest entry into the models folder: temp file, size and SHA-256 check, then moved into place.  Returns the path."""
    if not _valid_entry(entry, upd):
        raise DownloadError('Refused: not a valid model entry.')
    d = Path(folder)
    try:
        d.mkdir(parents=True, exist_ok=True)
        free = __import__('shutil').disk_usage(d).free
    except OSError as e:
        raise DownloadError(f'Cannot write to the models folder {d}: {e}') from e
    size = int(entry['size_bytes'])
    if free < size * 1.1:
        raise DownloadError(f'Not enough free disk space in {d} ({size / 1e6:.0f} MB needed).')
    part, final = d / f'.{entry["file"]}.part', d / entry['file']
    h = hashlib.sha256(); got = 0
    try:
        with _open(entry['url'], 30) as resp, open(part, 'wb') as out:
            while True:
                if cancel and cancel():
                    raise DownloadError('Download cancelled.')
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                got += len(chunk)
                if got > size * 1.01 + 1024:
                    raise DownloadError('The download is larger than expected; stopped.')
                out.write(chunk); h.update(chunk)
                if progress: progress(got, size)
        if got != size:
            raise DownloadError(f'Incomplete download ({got} of {size} bytes).')
        if h.hexdigest() != entry['sha256']:
            raise DownloadError('Checksum mismatch: the file was not installed. Try again later.')
        os.replace(part, final)
        return final
    except DownloadError:
        raise
    except Exception as e:
        raise DownloadError(f'Download failed ({type(e).__name__}: {e}).') from e
    finally:
        try:
            if part.exists():
                part.unlink()
        except OSError:
            pass


def describe_updates(res, spec):
    """Plain-text summary lines for a check_updates() result."""
    lines = []
    if res['script_newer']:
        lines.append(f"A newer script is available: {res['script_version']} (you have {VERSION}). {res['script_url']}")
    elif res['script_version']:
        lines.append(f'The script is up to date ({VERSION}).')
    for m, st in res['models']:
        if st == 'missing':
            lines.append(f"Available to download: {m['name']} ({m['size_bytes'] / 1e6:.0f} MB).")
        elif st == 'update':
            lines.append(f"Update available for {m['name']} (the installed file differs from the published one).")
    if not any(st != 'current' for _, st in res['models']):
        lines.append('Installed models are up to date.')
    return lines


def missing_model_message(spec, res=None):
    """Status text when no model is installed."""
    if res:
        d = next((m for m, st in res['models'] if st == 'missing' and m.get('default')), None) or next((m for m, st in res['models'] if st == 'missing'), None)
        if d:
            return f"No model installed. Press Download model to get {d['name']} ({d['size_bytes'] / 1e6:.0f} MB)."
    return f'No model installed. Press Download model, or put a .onnx file in {model_dir(spec)[0]}.'


# ----------------------------------------------------------------------------------------------- Siril I/O
def read_pixels(siril):
    data = siril.get_image_pixeldata()
    if data is None:
        fit = siril.get_image(with_pixels=True)
        data = fit.data if fit is not None else None
    if data is None:
        raise RuntimeError('No image loaded in Siril.')
    return np.array(data, copy=True)


def decode_pixels(raw):
    """Siril pixels -> float32 HxW or CxHxW (uint16 scaled to 0..1)."""
    raw = np.asarray(raw)
    if raw.dtype not in (np.dtype('float32'), np.dtype('uint16')):
        raise ValueError('Expected Siril float32 or uint16 pixels.')
    if raw.ndim == 2 or (raw.ndim == 3 and raw.shape[0] in (1, 3)):
        data = raw.astype(np.float32)
    else:
        raise ValueError('Only monochrome or channel-first RGB images are supported.')
    if raw.dtype == np.uint16:
        data /= 65535.0
    return np.ascontiguousarray(data), dict(dtype=raw.dtype, shape=raw.shape)


def encode_pixels(data, meta):
    if meta['dtype'] == np.uint16:
        out = np.rint(np.clip(data, 0, 1) * 65535).astype(np.uint16)
    else:
        out = np.asarray(data, np.float32)
    if out.shape != meta['shape']:
        raise ValueError('Output shape does not match the captured image.')
    return np.ascontiguousarray(out)


def fingerprint(raw):
    return raw.shape, raw.dtype.str, hashlib.sha256(np.ascontiguousarray(raw).tobytes()).digest()


def write_result(siril, data, meta, expected, label):
    """Apply to Siril with an Undo state, only if the image is still the one that was processed."""
    final = encode_pixels(data, meta)
    with siril.image_lock():
        current = read_pixels(siril)
        if fingerprint(current) != expected:
            raise RuntimeError('The image in Siril changed since Apply started. Nothing was written.')
        if siril.undo_save_state(label) is False:
            raise RuntimeError('Siril could not save Undo. Pixels were not overwritten.')
        if siril.set_image_pixeldata(final) is False:
            raise RuntimeError('Siril refused the modified pixels.')
        actual = read_pixels(siril)
        if actual.shape != final.shape or not np.allclose(actual, final, rtol=0, atol=0 if final.dtype == np.uint16 else 2e-7, equal_nan=True):
            raise RuntimeError('Siril readback differs from the result. Use Siril Undo if needed.')
    return fingerprint(actual)


def siril_header_text(siril):
    parts = []
    for fn in (lambda: siril.get_image_fits_header(return_as='str'), lambda: ' '.join(str(h) for h in (siril.get_image_history() or [])),
               lambda: siril.get_image_filename()):
        try:
            parts.append(str(fn()))
        except Exception:
            pass
    return ' '.join(parts)


def unique_path(p):
    p = Path(p)
    if not p.exists():
        return p
    for i in range(1, 1000):
        q = p.with_name(f'{p.stem}_{i}{p.suffix}')
        if not q.exists():
            return q
    raise FileExistsError(p)


def side_path(siril, suffix):
    """<image>_<suffix>.fit next to the image Siril has loaded (falls back to Siril's working folder)."""
    name = None
    try:
        name = siril.get_image_filename()
    except Exception:
        pass
    if name and Path(name).parent.is_dir() and str(Path(name).parent) != '.':
        return unique_path(Path(name).with_name(Path(name).stem.removesuffix('.fit') + f'_{suffix}.fit'))
    wd = None
    try:
        wd = siril.get_siril_wd()
    except Exception:
        pass
    return unique_path(Path(wd or Path.home()) / f'umbra_{suffix}.fit')


def write_fits(path, data, header=None, history=''):
    from astropy.io import fits
    hdr = header.copy() if header is not None else fits.Header()
    for k in ('BSCALE', 'BZERO', 'BLANK'):
        hdr.pop(k, None)
    if history:
        hdr['HISTORY'] = history
    fits.PrimaryHDU(np.asarray(data, np.float32), header=hdr).writeto(path)


def siril_fits_header(siril):
    try:
        from astropy.io import fits
        return fits.Header.fromstring(siril.get_image_fits_header(return_as='str'), sep='\n')
    except Exception:
        return None


def region_from_selection(siril, minimum=64):
    """Siril's current selection (x, y, w, h) when at least `minimum` px each way, else None."""
    try:
        sel = siril.get_siril_selection()
    except Exception:
        return None
    if sel and sel[2] >= minimum and sel[3] >= minimum:
        return tuple(int(v) for v in sel)
    return None


def crop_region(data, rect=None, size=768):
    """data (H,W) or (C,H,W) -> (cropped view copy, (x, y, w, h)).  rect given: that region; else a centred size x size crop (whole image if smaller)."""
    H, W = data.shape[-2:]
    if rect is None:
        w, h = min(size, W), min(size, H)
        rect = ((W - w) // 2, (H - h) // 2, w, h)
    x, y, w, h = rect
    x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    w, h = min(w, W - x), min(h, H - y)
    return np.ascontiguousarray(data[..., y:y + h, x:x + w]), (x, y, w, h)


# ----------------------------------------------------------------------------------------------- display (auto-stretch)
def _mtf(m, x):
    return (m - 1.0) * x / ((2.0 * m - 1.0) * x - m)


def stretch_params(img, target=0.25, shadow_k=-2.8):
    """Linked auto-stretch (one set of parameters for all channels), like an STF: (c0, midtone)."""
    v = np.asarray(img, np.float32)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 0.5, 1.0
    if v.size > 2_000_000:
        v = v[::max(1, v.size // 2_000_000)]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    c0 = min(max(0.0, med + shadow_k * mad), max(med - 1e-6, 0.0)) if mad > 0 else 0.0
    hi = float(np.percentile(v, 99.9)) if v.size else 1.0
    span = max(hi - c0, 1e-6) if hi > c0 else 1.0
    medn = min(max((med - c0) / span, 1e-6), 0.999)
    m = float(_mtf(target, medn))
    return c0, min(max(m, 1e-4), 0.9999), span


def apply_stretch(img, params):
    c0, m, span = params
    x = np.clip((np.nan_to_num(np.asarray(img, np.float32)) - c0) / span, 0, 1)
    y = _mtf(m, x)
    return np.clip(y, 0, 1)


def pool_for_display(img, max_side=1600):
    """Block-mean downsample (C,H,W) so the longer side is <= max_side."""
    H, W = img.shape[-2:]
    f = int(math.ceil(max(H, W) / max_side))
    if f <= 1:
        return img
    h, w = H // f * f, W // f * f
    a = np.nan_to_num(img[..., :h, :w]).reshape(*img.shape[:-2], h // f, f, w // f, f)
    return a.mean((-3, -1)).astype(np.float32)


def to_rgb8(img, params, flip=False):
    """(H,W) or (C,H,W) linear -> (H,W,3) uint8 with the given stretch."""
    a = np.asarray(img, np.float32)
    if a.ndim == 2:
        a = a[None]
    a = apply_stretch(a, params)
    if a.shape[0] == 1:
        a = np.repeat(a, 3, 0)
    rgb = np.ascontiguousarray((a.transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8))
    return rgb[::-1].copy() if flip else rgb


# ----------------------------------------------------------------------------------------------- UI
STYLE = """
QWidget { background: #242424; color: #dddddd; font-size: 10pt; }
QWidget#Card { background: #303030; border-radius: 10px; }
QLabel, QCheckBox, QSlider { background: transparent; }
QLabel#Title { font-size: 15pt; font-weight: bold; color: #dddddd; }
QLabel#Badge { color: #888888; font-size: 8pt; }
QLabel#Note { color: #999999; font-size: 9pt; }
QLabel#Value { color: #dddddd; font-weight: bold; min-width: 40px; }
QLabel#Caption { color: #aaaaaa; font-size: 9pt; }
QComboBox { background: #404040; border: 1px solid #666666; border-radius: 4px; padding: 3px 6px; }
QComboBox QAbstractItemView { background: #303030; selection-background-color: #1274d5; }
QPushButton { min-height: 20px; background: #505050; border: 1px solid #666666; border-radius: 5px; padding: 6px 12px; }
QPushButton:hover { background: #616161; }
QPushButton:checked { background: #1274d5; border-color: #338be7; color: #ffffff; }
QPushButton:disabled { color: #777777; border-color: #444444; }
QPushButton#Apply { min-height: 20px; background: #1274d5; color: #ffffff; font-weight: bold; border-color: #338be7; }
QPushButton#Apply:disabled { background: #354656; color: #808890; border-color: #43505c; }
QSlider::groove:horizontal { height: 4px; background: #505050; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #1474d5; border-radius: 2px; }
QSlider::handle:horizontal { background: #dddddd; width: 14px; margin: -6px 0; border-radius: 7px; }
QSlider::sub-page:horizontal:disabled { background: #555555; }
QProgressBar { border: 1px solid #444444; border-radius: 3px; max-height: 6px; }
QProgressBar::chunk { background: #1474d5; }
"""

IDLE_SIRIL = 'Set the controls, then press Preview or Apply. Apply replaces the image in Siril (Undo available).'
IDLE_TEST = 'Test mode (no Siril).'
_QT = {}


def _qt():
    """Build the Qt classes once (PyQt6 is imported only when a window is opened)."""
    if _QT:
        return _QT['ns']
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QProgressBar,
                                 QComboBox, QCheckBox, QSizePolicy, QStackedWidget, QMenu, QLayout)

    class Worker(QThread):
        done = pyqtSignal(object, object, float, str, object)
        failed = pyqtSignal(str, bool)       # message, during model load
        cancelled = pyqtSignal()
        prog = pyqtSignal(int, int)
        stage = pyqtSignal(str)

        def __init__(self, spec, model_path, data, params):
            super().__init__()
            self.spec, self.model_path, self.data, self.params = spec, model_path, data, params
            self.cancel_flag = False

        def run(self):
            loading = True
            try:
                t0 = time.time()
                if str(self.model_path) not in [c[0] for c in _ENGINE_CACHE.values()]:
                    self.stage.emit('Loading model…')
                eng = get_engine(self.spec, self.model_path)
                loading = False
                self.stage.emit('Running…')
                out, info = eng.run(self.data, self.params, progress=lambda n, t: self.prog.emit(int(n), int(t)), cancel=lambda: self.cancel_flag)
                self.done.emit(out, info, time.time() - t0, eng.provider, eng.note)
            except Exception as e:
                if not loading and str(e) == 'Cancelled':
                    self.cancelled.emit()
                    return
                import traceback
                tb = traceback.format_exc()
                write_error_log(self.spec, tb)
                self.failed.emit(f'{type(e).__name__}: {e}' if not loading else str(e), loading)

    class Task(QThread):
        done = pyqtSignal(object)
        failed = pyqtSignal(str)
        prog = pyqtSignal(int, int)

        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def run(self):
            try:
                self.done.emit(self.fn(lambda a, b: self.prog.emit(int(a), int(b))))
            except Exception as e:
                self.failed.emit(str(e) if isinstance(e, DownloadError) else f'{type(e).__name__}: {e}')

    def qimage(rgb):
        h, w = rgb.shape[:2]
        return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

    class PreviewWindow(QWidget):
        """Umbra <Tool> Preview (BETA): before/after with the same linked auto-stretch; hold Space (or the button) for the original."""
        def __init__(self, spec):
            super().__init__()
            self.spec = spec
            self.setWindowTitle(f'{spec.name} Preview (BETA)')
            self.setStyleSheet(STYLE)
            self.resize(820, 760)
            self.layers = {}
            self.pix = {}
            lay = QVBoxLayout(self)
            lay.setContentsMargins(8, 8, 8, 8)
            self.view = QLabel()
            self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.view.setMinimumSize(200, 200)
            self.view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            lay.addWidget(self.view, 1)
            row = QHBoxLayout()
            self.caption = QLabel('', objectName='Caption', wordWrap=True)
            row.addWidget(self.caption, 1)
            self.orig_btn = QPushButton('Show original')
            self.show_orig = False
            self.orig_btn.setToolTip('Hold the button (or the Space key) to see the original; release to go back to the result.')
            self.orig_btn.pressed.connect(lambda: self.hold(True))
            self.orig_btn.released.connect(lambda: self.hold(False))
            row.addWidget(self.orig_btn)
            self.alt_btn = None
            if spec.alt:
                self.alt_btn = QPushButton(spec.alt[0])
                self.alt_btn.setCheckable(True)
                self.alt_btn.toggled.connect(lambda c: (self.hold(False) if c else None, self.refresh()))
                row.addWidget(self.alt_btn)
            lay.addLayout(row)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        def hold(self, flag):
            self.show_orig = bool(flag)
            self.refresh()

        def show_result(self, before, after, alt, caption):
            """before / after / alt: (H,W,3) uint8 (alt may be None)."""
            self.pix = {'before': QPixmap.fromImage(qimage(before)), 'after': QPixmap.fromImage(qimage(after))}
            if alt is not None:
                self.pix['alt'] = QPixmap.fromImage(qimage(alt))
            if self.alt_btn is not None:
                self.alt_btn.setVisible(alt is not None)
                if alt is None:
                    self.alt_btn.setChecked(False)
            self.show_orig = False
            self.caption.setText(caption)
            self.refresh()
            self.show(); self.raise_()

        def current_key(self):
            if self.alt_btn is not None and self.alt_btn.isChecked() and 'alt' in self.pix:
                return 'alt'
            return 'before' if self.show_orig else 'after'

        def refresh(self):
            p = self.pix.get(self.current_key())
            if p is None or p.isNull():
                return
            self.view.setPixmap(p.scaled(self.view.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        def resizeEvent(self, e):
            super().resizeEvent(e)
            self.refresh()

        def keyPressEvent(self, e):
            if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
                self.hold(True)
            else:
                super().keyPressEvent(e)

        def keyReleaseEvent(self, e):
            if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
                self.hold(False)
            else:
                super().keyReleaseEvent(e)

    class Ui:
        """Control builders handed to the tool panels so every tool looks the same."""
        def __init__(self, view):
            self.view = view

        def slider(self, layout, text, lo, hi, default, fmt, tip=None):
            row = QHBoxLayout()
            label = QLabel(text)
            row.addWidget(label)
            val = QLabel(fmt(default), objectName='Value')
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(val)
            layout.addLayout(row)
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(lo, hi); s.setValue(default)
            s.valueChanged.connect(lambda v, l=val: l.setText(fmt(v)))
            if tip:
                label.setToolTip(tip); s.setToolTip(tip); val.setToolTip(tip)
            layout.addWidget(s)
            s.label_widget, s.value_widget = label, val
            return s

        def check(self, layout, text, tip=None, checked=False):
            c = QCheckBox(text)
            c.setChecked(checked)
            if tip:
                c.setToolTip(tip)
            layout.addWidget(c)
            return c

        def combo(self, layout, text, items, tip=None):
            row = QHBoxLayout()
            label = QLabel(text)
            row.addWidget(label)
            cb = QComboBox()
            for t, v in items:
                cb.addItem(t, v)
            if tip:
                label.setToolTip(tip); cb.setToolTip(tip)
            row.addWidget(cb, 1)
            layout.addLayout(row)
            cb.label_widget = label
            return cb

        def note(self, layout, text):
            l = QLabel(text, objectName='Note', wordWrap=True)
            layout.addWidget(l)
            return l

        @staticmethod
        def show(widget, flag):
            for w in (widget, getattr(widget, 'label_widget', None), getattr(widget, 'value_widget', None)):
                if w is not None:
                    w.setVisible(bool(flag))
            QTimer.singleShot(0, lambda: widget.window().adjustSize())

        @staticmethod
        def enable(widget, flag):
            for w in (widget, getattr(widget, 'label_widget', None), getattr(widget, 'value_widget', None)):
                if w is not None:
                    w.setEnabled(bool(flag))

        def model_dir(self):
            return model_dir(self.view.spec)[0]

    class ToolView(QWidget):
        """One tool: title, model card + tool controls, Preview / Apply, progress, status, Close. Used alone or as a page of the combined window."""
        def __init__(self, spec, siril=None, fake_image=None, prefs=None, upd=None, show_title=True, smoke=False):
            super().__init__()
            self.spec, self.siril, self.fake, self.prefs, self.smoke = spec, siril, fake_image, prefs, smoke
            self.upd = upd or updspec_for(spec)
            self.worker = self.net = self.preview = None
            self.info = None
            self.job = None
            self.cache = None            # (identity, params key, out, info) of the last whole-image preview (Gradient)
            self.last_preview = None     # (before, after, alt) uint8 of the last preview
            self.result = None
            self.last_error = None
            self.busy = False
            saved_model, saved_settings = (prefs.load(spec.key) if prefs else (None, {}))
            self.saved_model = saved_model
            lay = QVBoxLayout(self)
            lay.setContentsMargins(14, 12, 14, 12)
            lay.setSpacing(8)
            self.lay = lay
            if show_title:
                lay.addWidget(QLabel(spec.title, objectName='Title'))
                lay.addWidget(QLabel(f'v{VERSION}', objectName='Badge'))
            for n in spec.notes:
                lay.addWidget(QLabel(n, objectName='Note', wordWrap=True))
            card = QWidget(objectName='Card')
            card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            cl = QVBoxLayout(card)
            cl.setContentsMargins(12, 10, 12, 12)
            mrow = QHBoxLayout()
            mrow.addWidget(QLabel('Model'))
            self.model_box = QComboBox()
            mrow.addWidget(self.model_box, 1)
            cl.addLayout(mrow)
            brow = QHBoxLayout()
            self.dl_btn = QPushButton('Download model')
            self.dl_btn.setMinimumHeight(30)
            self.dl_btn.setToolTip("Download a model from this project's GitHub releases (checked against a SHA-256 before it is installed).")
            self.dl_btn.clicked.connect(self.show_dl_menu)
            self.upd_btn = QPushButton('Check for updates')
            self.upd_btn.setMinimumHeight(30)
            self.upd_btn.setToolTip('Looks for newer models and a newer script. Nothing is downloaded or replaced until you ask.')
            self.upd_btn.clicked.connect(lambda: self.fetch_info(manual=True))
            brow.addWidget(self.dl_btn); brow.addWidget(self.upd_btn)
            cl.addLayout(brow)
            self.upd_note = QLabel('', objectName='Note', wordWrap=True)
            self.upd_note.setOpenExternalLinks(True)
            self.upd_note.setTextFormat(Qt.TextFormat.RichText)
            self.upd_note.hide()
            cl.addWidget(self.upd_note)
            self.ui = Ui(self)
            self.panel = spec.build_panel(self.ui, cl, self)
            self.panel.set_settings(saved_settings)
            lay.addWidget(card)
            brow2 = QHBoxLayout()
            self.preview_btn = QPushButton('Preview')
            self.preview_btn.setToolTip('Runs on the selection in Siril (or a centred crop) and shows the result here. The image is not changed.' if not spec.preview_whole
                                        else 'Runs on the whole image and shows the result here, downsampled. The image is not changed.')
            self.preview_btn.clicked.connect(lambda: self.start('preview'))
            self.apply_btn = QPushButton('Apply', objectName='Apply')
            self.apply_btn.clicked.connect(lambda: self.start('apply'))
            brow2.addWidget(self.preview_btn); brow2.addWidget(self.apply_btn)
            lay.addLayout(brow2)
            prow = QHBoxLayout()
            self.progress = QProgressBar()
            self.progress.setRange(0, 0); self.progress.setTextVisible(False); self.progress.hide()
            self.cancel_btn = QPushButton('Cancel')
            self.cancel_btn.clicked.connect(self.cancel)
            self.cancel_btn.hide()
            prow.addWidget(self.progress, 1); prow.addWidget(self.cancel_btn)
            lay.addLayout(prow)
            self.idle_msg = IDLE_SIRIL if siril is not None else IDLE_TEST
            self.status = QLabel(self.idle_msg, objectName='Note', wordWrap=True)
            lay.addWidget(self.status)
            close = QPushButton('Close')
            close.clicked.connect(lambda: self.window().close())
            lay.addWidget(close)
            self.model_box.currentIndexChanged.connect(lambda *_: self.model_changed())
            self.folder_note = None
            self.refresh_models()

        # ---- status
        def set_status(self, msg, error=False):
            self.status.setText(msg)
            self.status.setStyleSheet('color: #e05050;' if error else '')
            if self.isVisible():                      # a wrapped status line may need more (or less) height
                QTimer.singleShot(0, self.window().adjustSize)

        def model_path(self):
            return self.model_box.currentData()

        def save_prefs(self):
            if self.prefs:
                self.prefs.save(self.spec.key, self.model_path(), self.panel.settings(), getattr(self.window(), 'current_key', None))

        def set_busy(self, flag, net=False):
            self.busy = flag
            has = bool(self.model_path())
            self.preview_btn.setEnabled(not flag and has); self.apply_btn.setEnabled(not flag and has)
            self.dl_btn.setEnabled(not flag); self.upd_btn.setEnabled(not flag)
            self.model_box.setEnabled(not flag)
            self.progress.setVisible(flag); self.cancel_btn.setVisible(flag and not net)
            if flag:
                self.progress.setRange(0, 0)

        # ---- models
        def refresh_models(self, select=None):
            sp = self.spec
            cur = select or self.model_path() or default_model_path(sp, self.saved_model)
            self.model_box.blockSignals(True)
            self.model_box.clear()
            items = list_models(sp)
            for label, path in items:
                self.model_box.addItem(label, path)
            i = self.model_box.findData(cur) if cur else -1
            if i < 0 and cur and Path(cur).is_file():
                self.model_box.addItem(model_label(sp, Path(cur).stem), cur)
                i = self.model_box.count() - 1
            if self.model_box.count():
                self.model_box.setCurrentIndex(max(i, 0))
            self.model_box.blockSignals(False)
            has = bool(self.model_path())
            self.preview_btn.setEnabled(has and not self.busy); self.apply_btn.setEnabled(has and not self.busy)
            self.dl_btn.setStyleSheet('' if has else 'background: #1274d5; color: #ffffff; font-weight: bold;')
            d, fb = model_dir(sp)
            if fb and self.folder_note is None:
                self.folder_note = f"Using models from the old '{sp.fallback_folder}' folder."
            self.model_changed(remember=bool(select))

        def model_changed(self, *_, remember=False):
            path = self.model_path()
            if not path:
                self.panel.model_changed(None)
                self.set_status(missing_model_message(self.spec, self.info))
                return
            self.panel.model_changed(path)
            if remember:
                self.save_prefs()
            idle = self.idle_msg
            if self.folder_note:
                idle = self.folder_note + ' ' + idle
            self.set_status(idle)

        # ---- updates / downloads (a button press starts every network action, except the manifest lookup at launch)
        def run_net(self, fn, on_done, busy_text):
            if self.net is not None and self.net.isRunning():
                return
            self.net = Task(fn)
            self.net.prog.connect(self.net_progress)
            self.net.done.connect(lambda r: (self.set_busy(False), on_done(r)))
            self.net.failed.connect(lambda m: (self.set_busy(False), self.net_failed(m)))
            self.set_busy(True, net=True)
            self.set_status(busy_text)
            self.net.start()

        def net_progress(self, got, total):
            if total > 0:
                self.progress.setRange(0, 1000); self.progress.setValue(int(1000 * got / total))
                self.set_status(f'Downloading… {got / 1e6:.0f} / {total / 1e6:.0f} MB')

        def net_failed(self, msg):
            self.set_status(msg + ('' if self.model_path() else ' No model is installed yet.'), True)

        def fetch_info(self, manual=False, then=None):
            sp = self.spec
            if offline_mode(sp):
                self.upd_note.setText(f'Offline mode ({sp.prefix}OFFLINE is set); using the installed models.'); self.upd_note.show()
                return
            def work(prog):
                return check_updates(fetch_manifest(self.upd, use_cache=not manual), sp)
            def done(res):
                self.info = res
                if manual:
                    from html import escape
                    lines = describe_updates(res, sp)
                    url = res['script_url']
                    html = '<br>'.join(escape(l).replace(escape(url), f'<a href="{escape(url)}">release page</a>') if url else escape(l) for l in lines)
                    self.upd_note.setText(html); self.upd_note.show()
                if then:
                    then()
                elif not self.model_path():
                    self.set_status(missing_model_message(sp, res))
            def failed(msg):
                self.info = None
                if manual or not self.model_path():
                    self.upd_note.setText(msg); self.upd_note.show()
                self.set_status(msg if not self.model_path() else self.idle_msg, not self.model_path())
            self.net = Task(work)
            self.net.done.connect(lambda r: (self.set_busy(False), done(r)))
            self.net.failed.connect(lambda m: (self.set_busy(False), failed(m)))
            self.set_busy(True, net=True)
            self.set_status('Checking the model list…')
            self.net.start()

        def show_dl_menu(self):
            if self.info is None:
                if offline_mode(self.spec):
                    self.set_status(f'Offline mode ({self.spec.prefix}OFFLINE is set). Put a .onnx file in {download_dir(self.spec)}.')
                else:
                    self.fetch_info(manual=False, then=self.show_dl_menu)
                return
            menu = QMenu(self)
            for m, st in self.info['models']:
                if st == 'current':
                    continue
                tag = 'Update ' if st == 'update' else 'Download '
                menu.addAction(f"{tag}{m['name']} ({m['size_bytes'] / 1e6:.0f} MB)").triggered.connect(lambda _=False, e=m: self.start_download(e))
            if menu.isEmpty():
                self.set_status('All published models are already installed and up to date.')
                return
            menu.exec(self.dl_btn.mapToGlobal(self.dl_btn.rect().bottomLeft()))

        def start_download(self, entry):
            def work(prog):
                return download_model(entry, self.upd, download_dir(self.spec), progress=prog)
            def done(path):
                self.info = None
                self.refresh_models(select=str(path))
                self.set_status(f"Installed {entry['name']}. " + self.idle_msg)
            self.run_net(work, done, f"Downloading {entry['name']}…")

        # ---- run
        def cancel(self):
            if self.worker is not None and self.worker.isRunning():
                self.worker.cancel_flag = True
                self.cancel_btn.setEnabled(False)

        def read_image(self, kind):
            """-> (data, meta, identity, header_text, region note, rect)"""
            sp = self.spec
            if self.siril is not None:
                with self.siril.image_lock():
                    raw = read_pixels(self.siril)
                    data, meta = decode_pixels(raw)
                    identity = fingerprint(raw)
                    htxt = siril_header_text(self.siril)
            else:
                data, meta, identity, htxt = self.fake, dict(dtype=np.dtype('float32'), shape=self.fake.shape), fingerprint(self.fake), ''
            return data, meta, identity, htxt

        def start(self, kind):
            if self.busy:
                return
            sp = self.spec
            path = self.model_path()
            if not path:
                self.set_status(missing_model_message(sp, self.info), True)
                return
            try:
                data, meta, identity, htxt = self.read_image(kind)
            except Exception as e:
                self.set_status(f'Could not read the image: {e}', True)
                return
            params = dict(self.panel.params())
            params.update(sp.image_params(htxt))
            rect, region, crop = None, 'whole image', None
            full = data
            if kind == 'preview':
                params = sp.preview_params(params)
                if not sp.preview_whole:
                    sel = region_from_selection(self.siril) if self.siril is not None else None
                    H, W = full.shape[-2:]
                    # Siril selections are in display coordinates; pixel rows are bottom-up (rows H-y-h .. H-y)
                    _, arect = crop_region(full, None if sel is None else (sel[0], H - sel[1] - sel[3], sel[2], sel[3]))
                    rect = sel if sel is not None else arect
                    x, y, w, h = arect
                    data = np.ascontiguousarray(full[..., y:y + h, x:x + w])
                    if w < 512 or h < 512:                   # fixed 512 px models: give small regions context, crop back afterwards
                        tw, th = min(max(w, 640), W), min(max(h, 640), H)
                        x0 = min(max(x - (tw - w) // 2, 0), W - tw); y0 = min(max(y - (th - h) // 2, 0), H - th)
                        if (tw, th) != (w, h):
                            data = np.ascontiguousarray(full[..., y0:y0 + th, x0:x0 + tw])
                            crop = (x - x0, y - y0, w, h)
                    region = (f'selection {rect[2]}×{rect[3]} px at {rect[0]},{rect[1]}' if sel is not None
                              else (f'centre crop {rect[2]}×{rect[3]} px' if (rect[2], rect[3]) != (W, H) else f'whole image {W}×{H} px'))
                else:
                    region = f'whole image {data.shape[-1]}×{data.shape[-2]} px'
            key = json.dumps(params, sort_keys=True, default=str) + str(path)
            if kind == 'apply' and sp.reuse_preview and self.cache and identity is not None and self.cache[0] == identity and self.cache[1] == key:
                self.job = dict(kind=kind, meta=meta, identity=identity, params=params, region=region, data=data, key=key, htxt=htxt)
                self.finish(self.cache[2], self.cache[3], 0.0, self.cache[4], self.cache[5], reused=True)
                return
            if sp.linear_only:
                w = stretched_warning(data)
                if w:
                    msg = f'{sp.name}: Warning: {w}'
                    if self.siril is not None:
                        try:
                            self.siril.log(msg)
                        except Exception:
                            pass
                    else:
                        print(msg, flush=True)
            self.job = dict(kind=kind, meta=meta, identity=identity, params=params, region=region, data=data, key=key, htxt=htxt, rect=rect, crop=crop)
            self.result = None; self.last_error = None
            self.save_prefs()
            self.set_busy(True)
            self.cancel_btn.setEnabled(True)
            self.set_status('Loading model…' if str(path) not in [c[0] for c in _ENGINE_CACHE.values()] else 'Running…')
            self.worker = Worker(sp, path, data, params)
            self.worker.stage.connect(self.set_status)
            self.worker.prog.connect(self.on_progress)
            self.worker.done.connect(self.finish)
            self.worker.failed.connect(self.fail)
            self.worker.cancelled.connect(self.was_cancelled)
            self.worker.start()

        def on_progress(self, n, total):
            if total > 0:
                self.progress.setRange(0, total); self.progress.setValue(n)
                self.set_status(f'Running… {self.spec.progress_unit} {n} / {total}')

        def was_cancelled(self):
            self.set_busy(False)
            self.set_status('Cancelled. The image was not changed.')

        def fail(self, msg, loading=False):
            self.set_busy(False)
            self.last_error = msg
            self.set_status((f'Could not load the model: {msg}' if loading else msg), True)

        def finish(self, out, info, secs, provider, note, reused=False):
            sp, job = self.spec, self.job
            self.set_busy(False)
            extra = ' '.join(dict.fromkeys(info.get('notes') or []))
            if note and note not in extra:
                extra = (extra + ' ' + note).strip()
            try:
                if job['kind'] == 'preview':
                    if job.get('crop'):                       # run on a larger context region; show only the selected part
                        cx, cy, cw, ch = job['crop']
                        sl = (Ellipsis, slice(cy, cy + ch), slice(cx, cx + cw))
                        out = out[sl]
                        job = dict(job, data=job['data'][sl])
                        info = dict(info, **{k: np.asarray(v)[sl] for k, v in info.items() if isinstance(v, np.ndarray) and v.ndim >= 2})
                    self.show_preview(out, info, job)
                    if sp.reuse_preview and job['identity'] is not None:
                        self.cache = (job['identity'], job['key'], out, info, provider, note)
                    self.result = out
                    self.set_status(f'Preview done in {secs:.1f} s. Ran on {provider}. The image was not changed.' + (' ' + extra if extra else ''))
                    return
                label = f"{sp.name} {VERSION} {sp.label(job['params'])}"
                write_out = sp.output(job['params'], out, info)
                msg = ''
                ctx = types.SimpleNamespace(siril=self.siril, header_text=job['htxt'], history=label, view=self)
                if self.siril is not None:
                    self.cache = None
                    self.job['identity'] = write_result(self.siril, write_out, job['meta'], job['identity'], label)
                    msg = sp.post_apply(ctx, job['params'], info) or ''
                    self.siril.log(f'{label} done in {secs:.1f} s (Undo available).' + (' ' + msg if msg else ''))
                H0, W0 = job['data'].shape[-2:]
                if not sp.preview_whole and min(H0, W0) < 512:
                    extra = (extra + ' Images smaller than 512 px give less accurate results near the edges.').strip()
                self.result = write_out
                self.set_status(f'Done in {secs:.1f} s. Ran on {provider}. ' + ('Applied to Siril; use Siril Undo to revert.' if self.siril is not None else 'Test mode: nothing was written.')
                                + (' ' + msg if msg else '') + (' ' + extra if extra else ''))
            except Exception as e:
                self.fail(str(e))

        def show_preview(self, out, info, job):
            sp = self.spec
            before = pool_for_display(job['data'])
            after = pool_for_display(out)
            flip = self.siril is not None
            sp_before = stretch_params(before)
            b8, a8 = to_rgb8(before, sp_before, flip), to_rgb8(after, sp_before, flip)
            alt8 = None
            if sp.alt and info.get(sp.alt[1]) is not None:
                alt = pool_for_display(np.asarray(info[sp.alt[1]], np.float32))
                alt8 = to_rgb8(alt, stretch_params(alt), flip)
            self.last_preview = (b8, a8, alt8)
            if self.preview is None:
                self.preview = PreviewWindow(sp)
            self.preview.show_result(b8, a8, alt8, f'Preview of the {job["region"]}. Same auto-stretch for before and after.')

        def shutdown(self):
            for t in (self.worker, self.net):
                if t is not None and t.isRunning():
                    if t is self.worker:
                        t.cancel_flag = True
                    t.wait(8000)
            if self.preview is not None:
                self.preview.close()

    class MainWindow(QWidget):
        def __init__(self, specs, combined, siril=None, fake_image=None, smoke=False):
            super().__init__()
            self.specs = {s.key: s for s in specs}
            self.combined = combined
            self.setStyleSheet(STYLE)
            self.setFixedWidth(380)
            self.views = {}
            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)
            lay.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)   # window height always follows its content (wrapped notes, hidden rows)
            if combined:
                self.setWindowTitle('Umbra Astro AI (BETA)')
                prefs = Prefs(SCRIPT_DIR / 'Umbra_Astro_AI_prefs.json', combined=True, enabled=not smoke)
                top = QWidget()
                tl = QVBoxLayout(top)
                tl.setContentsMargins(14, 12, 14, 0)
                tl.setSpacing(4)
                tl.addWidget(QLabel('Umbra Astro AI (BETA)', objectName='Title'))
                tl.addWidget(QLabel(f'v{VERSION}', objectName='Badge'))
                row = QHBoxLayout()
                row.addWidget(QLabel('Tool'))
                self.tool_box = QComboBox()
                for s in specs:
                    self.tool_box.addItem(s.name, s.key)
                row.addWidget(self.tool_box, 1)
                tl.addLayout(row)
                lay.addWidget(top)
                self.stack = None
                for s in specs:
                    v = ToolView(s, siril, fake_image, prefs, combined_upd(specs),
                                 show_title=False, smoke=smoke)
                    self.views[s.key] = v
                    lay.addWidget(v)
                    v.hide()
                self.prefs = prefs
                first = prefs.load_tool()
                self.current_key = first if first in self.views else specs[0].key
                self.tool_box.setCurrentIndex(self.tool_box.findData(self.current_key))
                self.tool_box.currentIndexChanged.connect(lambda *_: self.switch(self.tool_box.currentData()))
                self.switch(self.current_key, initial=True)
            else:
                s = specs[0]
                self.setWindowTitle(s.title)
                self.current_key = s.key
                v = ToolView(s, siril, fake_image, Prefs(SCRIPT_DIR / f'{s.script}_prefs.json', enabled=not smoke), show_title=True, smoke=smoke)
                self.views[s.key] = v
                lay.addWidget(v)

        def view(self):
            return self.views[self.current_key]

        def refit(self):
            if self.layout() is not None:
                self.layout().activate()
            self.adjustSize()

        def switch(self, key, initial=False):
            if not initial:
                self.view().save_prefs()
                unload_engines()
            self.current_key = key
            for k, v in self.views.items():           # only the current page is visible, so the window is exactly as tall as that page needs
                v.setVisible(k == key)
            self.refit()
            QTimer.singleShot(0, self.refit)
            self.view().save_prefs() if not initial else None

        def closeEvent(self, e):
            for v in self.views.values():
                v.shutdown()
            try:
                self.view().save_prefs()
            except Exception:
                pass
            super().closeEvent(e)

    _QT['ns'] = types.SimpleNamespace(**{k: v for k, v in locals().items() if k in (
        'Worker', 'Task', 'PreviewWindow', 'Ui', 'ToolView', 'MainWindow', 'QApplication', 'QTimer')})
    return _QT['ns']


def fake_image(seed=0, size=192):
    """Synthetic linear RGB frame (noise + a few stars + a gentle gradient) for --ui-test."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    img = (0.02 + 0.003 * rng.standard_normal((3, size, size)) + (0.01 * xx / size)[None]).astype(np.float32)
    for cy, cx in ((40, 50), (110, 120), (150, 40), (70, 150)):
        img += (0.3 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.6 ** 2)))[None].astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32)


def run_app(specs, combined, siril=None, fake=None, smoke=False, shots=None):
    """Open the window. smoke: run Preview once for every tool on the fake image, report, exit (return code 0 = all fine)."""
    q = _qt()
    app = q.QApplication.instance() or q.QApplication(sys.argv)
    win = q.MainWindow(specs, combined, siril, fake, smoke)
    win.show()
    if not smoke:
        for v in win.views.values():
            if v is win.view():
                q.QTimer.singleShot(200, lambda v=v: (v.fetch_info(manual=False) if not offline_mode(v.spec) else None))
        if combined:
            win.tool_box.currentIndexChanged.connect(lambda *_: (win.view().info is None and not offline_mode(win.view().spec)) and win.view().fetch_info(manual=False))
        return app.exec()
    state = dict(i=0, ok=True, t0=time.time())
    keys = [s.key for s in specs]
    outdir = Path(shots) if shots else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    def snap(v, key):
        if not outdir:
            return
        pre = f'Umbra_Astro_AI_{key}' if combined else win.view().spec.script
        win.grab().save(str(outdir / f'{pre}_panel.png'))
        if v.preview is not None and v.preview.isVisible():
            v.preview.grab().save(str(outdir / f'{pre}_preview.png'))
            if v.preview.alt_btn is not None and v.preview.alt_btn.isVisible():
                v.preview.alt_btn.setChecked(True)
                v.preview.grab().save(str(outdir / f'{pre}_preview_{v.spec.alt[1]}.png'))
                v.preview.alt_btn.setChecked(False)

    def next_tool():
        if state['i'] >= len(keys):
            print('UI TEST', 'OK' if state['ok'] else 'FAILED', flush=True)
            win.close(); app.quit()
            return
        key = keys[state['i']]
        if combined:
            win.tool_box.setCurrentIndex(win.tool_box.findData(key))
        v = win.view()
        if not v.model_path():
            # no model: Preview/Apply must be disabled and the status must say so
            good = (not v.preview_btn.isEnabled() and not v.apply_btn.isEnabled() and 'No model installed' in v.status.text())
            print(f'UI TEST {key}: no model installed; controls {"OK" if good else "BAD"}: {v.status.text()}', flush=True)
            state['ok'] &= good
            app.processEvents(); snap(v, key)
            state['i'] += 1
            q.QTimer.singleShot(50, next_tool)
            return
        v.start('preview')
        state['t0'] = time.time()
        q.QTimer.singleShot(250, lambda: poll(v, key))

    def poll(v, key):
        if v.busy and time.time() - state['t0'] < 900:
            q.QTimer.singleShot(250, lambda: poll(v, key))
            return
        ok = v.last_error is None and v.result is not None and bool(np.isfinite(v.result).all()) and v.last_preview is not None
        if ok:
            v.preview.orig_btn.pressed.emit(); ok &= v.preview.current_key() == 'before'; v.preview.orig_btn.released.emit(); ok &= v.preview.current_key() == 'after'
        print(f'UI TEST {key}: {"OK" if ok else "FAILED"} {v.status.text()}', flush=True)
        state['ok'] &= ok
        app.processEvents()
        snap(v, key)
        if v.preview is not None:
            v.preview.close()
        state['i'] += 1
        q.QTimer.singleShot(50, next_tool)
    q.QTimer.singleShot(100, next_tool)
    app.exec()
    return 0 if state['ok'] else 1


# ----------------------------------------------------------------------------------------------- CLI
def read_fits(path):
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        hdu = next(x for x in h if x.data is not None)
        data = np.asarray(hdu.data, np.float32)
        header = hdu.header.copy()
        text = ' '.join(repr(x.header) for x in h) + ' ' + ' '.join(str(x.name) for x in h) + ' ' + Path(path).name
    return data, header, text


def run_file(spec, a):
    src, dst = Path(a.image).resolve(), Path(a.out).resolve()
    if dst == src:
        raise SystemExit('Refusing to overwrite the input file (--out equals --image).')
    if dst.exists():
        raise SystemExit(f'Output exists, refusing to overwrite: {dst}')
    data, header, text = read_fits(src)
    params = spec.params_from_args(a)
    params.update(spec.image_params(text))
    if spec.linear_only:
        w = stretched_warning(data)
        if w:
            print(f'{spec.name}: Warning: {w}', flush=True)
    path = a.model or default_model_path(spec)
    if not path:
        raise SystemExit(f'No model installed. Run with --download, or put a .onnx file in {model_dir(spec)[0]}.')
    t0 = time.time()
    eng = get_engine(spec, path)
    out, info = eng.run(data, params, progress=None, cancel=None)
    label = f"{spec.name} {VERSION} {spec.label(params)}"
    write_out = spec.output(params, out, info)
    for k in ('BSCALE', 'BZERO', 'BLANK'):
        header.pop(k, None)
    header['HISTORY'] = label
    from astropy.io import fits
    fits.PrimaryHDU(np.asarray(write_out, np.float32), header=header).writeto(dst)
    msg = spec.post_cli(a, dst, info, header, label)
    print(f'wrote {dst} shape={write_out.shape} finite={bool(np.isfinite(write_out).all())} device={eng.provider} model={Path(path).name} '
          f'{time.time() - t0:.1f} s' + (f' {msg}' if msg else '') + (f' NOTE: {" ".join(info.get("notes") or [])}' if info.get('notes') else '')
          + (f' NOTE: {eng.note}' if eng.note else ''))


def cli_updates(specs, a, upd_for=None):
    """--check-updates / --download for one or several tools. Returns an exit code."""
    try:
        for spec in specs:
            upd = upd_for(spec) if upd_for else updspec_for(spec)
            if offline_mode(spec):
                print(f'Offline mode ({spec.prefix}OFFLINE is set); nothing to check.')
                continue
            res = check_updates(fetch_manifest(upd), spec)
            if len(specs) > 1:
                print(f'== {spec.name}')
            if a.check_updates:
                print('\n'.join(describe_updates(res, spec)))
            if a.download:
                want = [e for e, st in res['models'] if st != 'current' and (a.download == 'all' or (a.download == 'default' and e.get('default')) or a.download == e['id'])]
                if not want and len(specs) == 1:
                    print('Nothing to download (installed and up to date, or unknown model id).')
                for e in want:
                    last = [0]
                    def pr(g, t, e=e, last=last):
                        if g - last[0] >= 20_000_000 or g == t:
                            last[0] = g; print(f"  {e['name']}: {g / 1e6:.0f} / {t / 1e6:.0f} MB", flush=True)
                    print('installed', download_model(e, upd, download_dir(spec), pr))
        return 0
    except DownloadError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1


def main(specs, combined=False, doc=''):
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', help='standalone: FITS file to process (never modified)')
    ap.add_argument('--out', help='standalone: output FITS (must not exist)')
    ap.add_argument('--model', help='model file (*.onnx); default: the remembered / first model')
    if combined:
        ap.add_argument('--tool', choices=[s.key for s in specs], help='which tool to run / check (default for --check-updates and --download: all)')
    for s in specs:
        s.add_cli(ap)
    ap.add_argument('--check-updates', action='store_true', help='compare installed models and this script with the published manifest, then exit')
    ap.add_argument('--download', nargs='?', const='default', metavar='MODEL_ID|all', help='download a published model (default: the default model) into the models folder, then exit')
    ap.add_argument('--ui-test', action='store_true', help='open the window on a fake image, run Preview once, exit')
    ap.add_argument('--screenshots', help=argparse.SUPPRESS)
    a = ap.parse_args()
    chosen = [s for s in specs if not combined or not a.tool or s.key == a.tool]
    if a.check_updates or a.download:
        if combined:
            return cli_updates(chosen, a, lambda s: combined_upd(specs))
        return cli_updates(chosen, a)
    if a.ui_test:
        return run_app(specs, combined, None, fake_image(), smoke=True, shots=a.screenshots)
    if a.image:
        if not a.out:
            ap.error('--out is required with --image')
        spec = chosen[0] if not combined else (TOOLS.get(a.tool) if a.tool else None)
        if spec is None:
            ap.error('--tool is required with --image in the combined script')
        spec.check_cli(a, ap)
        run_file(spec, a)
        return 0
    try:
        import_engine(specs[0])   # import onnxruntime before Qt starts; any failure is kept and shown in the window
    except Exception:
        pass
    import sirilpy as s
    siril = s.SirilInterface()
    try:
        siril.connect()
        siril.cmd('requires', '1.4.0')
        return run_app(specs, combined, siril)
    finally:
        siril.disconnect()


# ============================================================================================ engines
def _engine_starless():
    """Umbra Starless engine: pure numpy + scipy + onnxruntime port of the Umbra Starless v0.0.4 pipeline (Remove Stars, then Apply
    with the default settings).  No torch, no Qt, no Siril.

    Pipeline (same order and maths as the PyTorch script):
      1. data mode: auto (is_stretched heuristic) / linear / stretched
      2. halo pass: the network runs on a 1/4-scale copy; only the extended glow of very bright saturated stars is kept (grey opening,
         seeds = big saturated cores) and subtracted before the main pass (large_halos)
      3. main pass: tiled network (fixed 512 x 512 tiles, smaller images reflect-padded to 512, 64 px cosine-blended overlap) on x - halo; whole-image normalisation (median offset,
         Haar noise sigma, asinh compression K=8); S = min(S, x)
      4. clean_residue + noise_fill: the starless image is the original outside star footprints; inside, the network answer is shifted
         to its surroundings, clamped to its local median +- tolerance, and grain-matched noise is added
      5. star layer K: linear  x - S;  stretched  unscreen 1 - (1 - x) / (1 - S)  (so S + K, or screen(S, K), gives back x)
      6. Reduce Stars: amount 1 = S exactly, 0 = x exactly, between = star layer un-stretched by one smooth curve (stars shrink)

    NOT included (not part of the published version): correction marks (Remove / Keep painting) and the Auto-keep critic.
    NaN pixels: filled from the nearest valid pixel before processing, restored to NaN in every output.

    params (all optional):
      amount  float 0..1, default 1.0   1 = full star removal, <1 = reduce stars, 0 = original
      mode    'auto' | 'linear' | 'stretched', default 'auto'
      output  'starless' | 'stars', default 'starless'   ('stars' returns the star layer as the image)
    info: {'notes': [str], 'stars': star layer (same shape/dtype as data, NaN where data is NaN), 'mode_used': 'linear'|'stretched',
           'sigma': global noise sigma}
    """
    import math
    import numpy as np

    TOOL = 'starless'
    TILE = 512
    OVERLAP = 64
    K_COMP = 8.0
    HALO_FACTOR = 4
    A_SPAN = 1.5
    FADE0 = 0.8


    def probe_feed():
        rng = np.random.default_rng(1)
        return {'x': (rng.standard_normal((1, 3, TILE, TILE)) * 3).astype(np.float32),
                'cond': (rng.random((1, 1, TILE, TILE)) > 0.9).astype(np.float32)}


    # ------------------------------------------------------------------------------------------------ numpy / scipy replacements of torch ops
    def _resize_bilinear(a, H, W):
        """(N,h,w) -> (N,H,W), bilinear, align_corners=False, no antialias (same as torch F.interpolate)."""
        n, h, w = a.shape
        def idx(inn, out):
            src = np.maximum((np.arange(out, dtype=np.float64) + 0.5) * (inn / out) - 0.5, 0.0)
            i0 = np.minimum(np.floor(src).astype(np.int64), inn - 1)
            i1 = np.minimum(i0 + 1, inn - 1)
            return i0, i1, (src - i0).astype(np.float32)
        y0, y1, ly = idx(h, H)
        x0, x1, lx = idx(w, W)
        t = a[:, y0, :] * (1 - ly)[None, :, None] + a[:, y1, :] * ly[None, :, None]
        return (t[:, :, x0] * (1 - lx)[None, None, :] + t[:, :, x1] * lx[None, None, :]).astype(np.float32)


    def _gblur(a, s):
        """Gaussian blur of the last two axes (any leading dims).  Wide blurs run at 1/f scale like the torch version."""
        from scipy import ndimage
        a = np.ascontiguousarray(a, np.float32)
        sh = a.shape
        t = a.reshape(-1, *sh[-2:])
        f = int(s // 2.5)
        if f >= 2 and min(sh[-2:]) >= 8 * f:
            H, W = sh[-2:]
            ph, pw = (-H) % f, (-W) % f
            tp = np.pad(t, ((0, 0), (0, ph), (0, pw)), mode='edge') if (ph or pw) else t
            n = tp.shape[0]
            d = tp.reshape(n, tp.shape[1] // f, f, tp.shape[2] // f, f).mean((2, 4), dtype=np.float32)
            d = _gblur(d, math.sqrt(max(s * s - f * f / 12.0, 0.25)) / f)
            u = _resize_bilinear(d.reshape(n, *d.shape[-2:]), d.shape[-2] * f, d.shape[-1] * f)[:, :H, :W]
            return u.reshape(sh)
        r = max(1, int(math.ceil(3 * s)))
        k = np.exp(-0.5 * (np.arange(-r, r + 1, dtype=np.float32) / np.float32(s)) ** 2).astype(np.float32)
        k /= k.sum()
        t = ndimage.convolve1d(t, k, axis=2, mode='mirror')
        t = ndimage.convolve1d(t, k, axis=1, mode='mirror')
        return t.reshape(sh).astype(np.float32)


    def _lmedian(v):
        """lower median (what torch.median returns)."""
        v = np.asarray(v).ravel()
        return v[np.argpartition(v, (v.size - 1) // 2)[(v.size - 1) // 2]]


    # ------------------------------------------------------------------------------------------------ normalisation (ust/norm.py)
    def _sub(x, max_px=4_000_000):
        st = max(1, int((x.shape[-1] * x.shape[-2] / max_px) ** 0.5))
        return x[:, ::st, ::st]


    def haar_sigma(x):
        x = x[:, : x.shape[1] // 2 * 2, : x.shape[2] // 2 * 2].astype(np.float32)
        d = (x[:, 0::2, 0::2] - x[:, 1::2, 0::2] - x[:, 0::2, 1::2] + x[:, 1::2, 1::2]) / 2.0
        ok = np.isfinite(x[:, 0::2, 0::2]) & np.isfinite(x[:, 1::2, 1::2]) & (x[:, 0::2, 0::2] != 0) & np.isfinite(d)
        out = []
        for c in range(x.shape[0]):
            v = _sub(d[c:c + 1])[_sub(ok[c:c + 1])]
            if v.size > 100:
                med = _lmedian(v)
                out.append(np.float32(1.4826) * _lmedian(np.abs(v - med)))
            else:
                out.append(np.float32(0.0))
        return np.array(out, np.float32)


    def image_stats(x):
        xs = _sub(x.astype(np.float32))
        m = np.array([_lmedian(np.nan_to_num(xs[c][np.isfinite(xs[c])])) for c in range(x.shape[0])], np.float32)
        s = haar_sigma(x)
        rng = np.array([np.quantile(np.nan_to_num(xs[c]).ravel(), 0.999) for c in range(x.shape[0])], np.float32) - m
        return m, np.maximum(s, (1e-5 * np.maximum(np.abs(rng), 1e-12)).astype(np.float32))


    def _compress(xn):
        return (K_COMP * np.arcsinh(xn / K_COMP)).astype(np.float32)


    def _decompress(y):
        return (K_COMP * np.sinh(np.clip(y, -150, 150) / K_COMP)).astype(np.float32)


    # ------------------------------------------------------------------------------------------------ layout / data mode
    def is_stretched(x):
        s = x[..., ::4, ::4]
        med = float(np.median(s)); hi = float(np.percentile(s, 99.9))
        return med > 0.1 and med > 0.06 * max(hi, 1e-6)


    def to_chw(data):
        a = np.asarray(data, np.float32)
        if a.ndim == 2:
            return a[None].repeat(3, 0), '2d'
        if a.shape[0] in (1, 3):
            return (a.repeat(3, 0) if a.shape[0] == 1 else a), ('1hw' if a.shape[0] == 1 else 'chw')
        return np.moveaxis(a, -1, 0), 'hwc'


    def from_chw(a, lay):
        if lay == '2d':
            return a.mean(0)
        if lay == '1hw':
            return a.mean(0, keepdims=True)
        if lay == 'hwc':
            return np.moveaxis(a, 0, -1)
        return a


    def _fill_nan(x, bad):
        """Replace non-finite pixels (union over channels) by the nearest valid pixel."""
        from scipy import ndimage
        if bad.all():
            return np.zeros_like(x)
        _, (iy, ix) = ndimage.distance_transform_edt(bad, return_indices=True)
        return x[:, iy, ix]


    # ------------------------------------------------------------------------------------------------ the network, tiled
    def _tile_starts(n):
        return list(range(0, max(n - 2 * OVERLAP, 1), TILE - 2 * OVERLAP))


    def _n_tiles(H, W):
        return len(_tile_starts(H)) * len(_tile_starts(W))


    _PAD_FILL = {'mode': 'reflect'}


    def _pad_tile(a, ph, pw):
        """Pad a (3,h,w) tile to 512 x 512.  First reflect up to a multiple of 32 (what the torch script does), then fill the rest."""
        h, w = a.shape[-2:]
        p32h, p32w = (-h) % 32, (-w) % 32
        if p32h or p32w:
            a = np.pad(a, ((0, 0), (0, p32h), (0, p32w)), mode='reflect')
        rh, rw = ph - p32h, pw - p32w
        if rh or rw:
            m = _PAD_FILL['mode']
            a = np.pad(a, ((0, 0), (0, rh), (0, rw)), mode='reflect' if m == 'reflect' else ('edge' if m == 'edge' else 'constant'))
        return a


    def _starless_core(run_fn, x, tick, tile=TILE, overlap=OVERLAP):
        """x: (3,H,W) float32 -> (S in x's units, global noise sigma).  Tiled, cosine-blended overlap."""
        C, H, W = x.shape
        xt = np.ascontiguousarray(np.nan_to_num(x), np.float32)
        m, s = image_stats(xt)
        xc = _compress((xt - m[:, None, None]) / s[:, None, None])
        step = tile - 2 * overlap
        ys = range(0, max(H - 2 * overlap, 1), step); xs = range(0, max(W - 2 * overlap, 1), step)
        acc = np.zeros((3, H, W), np.float32); wacc = np.zeros((1, H, W), np.float32)
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0, math.pi, overlap + 2, dtype=np.float32)[1:-1])).astype(np.float32)
        w1 = np.ones(tile, np.float32); w1[:overlap] = ramp; w1[-overlap:] = ramp[::-1]
        wt = (w1[:, None] * w1[None, :])[None]
        for y in ys:
            for x_ in xs:
                y0, x0 = min(y, max(H - tile, 0)), min(x_, max(W - tile, 0))
                a = xc[:, y0:y0 + tile, x0:x0 + tile]
                h, w = a.shape[-2:]
                ph, pw = tile - h, tile - w                  # the ONNX models are fixed at 512 x 512: always run exactly one full tile
                if ph or pw:
                    a = _pad_tile(a, ph, pw)
                feed = {'x': np.ascontiguousarray(a[None], np.float32), 'cond': np.zeros((1, 1, tile, tile), np.float32)}
                r = np.asarray(run_fn(feed)[0], np.float32)[0][:, :h, :w]
                z = _decompress(a[:, :h, :w] + r)
                ww = wt[:, :h, :w]
                acc[:, y0:y0 + h, x0:x0 + w] += z * ww
                wacc[:, y0:y0 + h, x0:x0 + w] += ww
                tick()
        zn = acc / np.maximum(wacc, 1e-6)
        return (zn * s[:, None, None] + m[:, None, None]).astype(np.float32), float(s.mean())


    def _large_halos(run_fn, x, tick, factor=HALO_FACTOR):
        """Extended glow of very bright saturated stars, to subtract before the main pass (default settings of the script)."""
        from scipy import ndimage
        C, H, W = x.shape
        h2, w2 = H // factor, W // factor
        if min(h2, w2) < 64:
            return np.zeros_like(x)
        xs = x[:, :h2 * factor, :w2 * factor].reshape(C, h2, factor, w2, factor).mean((2, 4), dtype=np.float32)
        Sd, sgd = _starless_core(run_fn, xs, tick)
        R = np.maximum(xs - Sd, 0)
        yy, xx = np.mgrid[-4:5, -4:5]; disc = (yy * yy + xx * xx) <= 16
        Rl = np.stack([ndimage.grey_opening(R[c], footprint=disc) for c in range(R.shape[0])])
        Rl = np.minimum(Rl, _gblur(Rl, 1.5))
        xl = x.mean(0); xmax = float(np.percentile(xl, 99.999))
        sat = (xl > 0.85 * xmax) | (xl - float(np.median(xl)) > 5000 * sgd * factor)
        sl, ns = ndimage.label(sat)
        if ns:
            area = ndimage.sum(sat, sl, np.arange(1, ns + 1))
            sat = np.isin(sl, np.nonzero(area >= 150)[0] + 1)
        if not sat.any():
            return np.zeros_like(x)
        seed = sat[:h2 * factor, :w2 * factor].reshape(h2, factor, w2, factor).any((1, 3))
        lab, _ = ndimage.label(_gblur(Rl.mean(0), 1.0) > 2 * sgd)
        ids = np.unique(lab[seed]); ids = ids[ids > 0]
        if len(ids) == 0:
            return np.zeros_like(x)
        Rl = Rl * np.clip(_gblur(np.isin(lab, ids).astype(np.float32), 2.0) * 1.3, 0, 1)[None]
        up = _resize_bilinear(Rl, h2 * factor, w2 * factor)
        out = np.zeros_like(x); out[:, :h2 * factor, :w2 * factor] = up
        return _gblur(out, factor / 2).astype(np.float32)


    def _starless(run_fn, x, tick):
        """halo pass, then the network at full resolution on x - halo.  Returns (S, sigma)."""
        Hh = _large_halos(run_fn, x, tick)
        S, sg = _starless_core(run_fn, x - Hh, tick)
        return np.minimum(S, x), sg


    # ------------------------------------------------------------------------------------------------ residue cleaning and noise fill
    def clean_residue(x, S, sigma, k=0.6, half_size=6, dil=3):
        from scipy import ndimage
        C, H, W = S.shape
        rem = _gblur((x - S).mean(0), 1.0) > 3.0 * sigma
        Fp = ndimage.binary_dilation(rem, iterations=dil)
        out_w = (~Fp).astype(np.float32)
        bias = _gblur((x - S) * out_w[None], 12.0) / np.maximum(_gblur(out_w, 12.0), 1e-3)[None]
        bias = np.where(_gblur(out_w, 12.0)[None] > 0.02, bias, 0)
        S1 = S + bias
        h2, w2 = H // 2, W // 2
        Sd = S1[:, :h2 * 2, :w2 * 2].reshape(C, h2, 2, w2, 2).mean((2, 4), dtype=np.float32)
        md = np.stack([ndimage.median_filter(Sd[c], size=half_size) for c in range(C)])
        med = _resize_bilinear(np.ascontiguousarray(md, np.float32), h2 * 2, w2 * 2)
        if med.shape[-2:] != S.shape[-2:]:
            med = np.pad(med, ((0, 0), (0, H - h2 * 2), (0, W - w2 * 2)), mode='edge')
        dev = np.abs(Sd.mean(0) - md.mean(0))
        tex = 1.4826 * ndimage.median_filter(dev, size=2 * half_size - 1)
        tex = _resize_bilinear(np.ascontiguousarray(tex, np.float32)[None], h2 * 2, w2 * 2)[0]
        if tex.shape != (H, W):
            tex = np.pad(tex, ((0, H - h2 * 2), (0, W - w2 * 2)), mode='edge')
        tol = np.maximum(k * sigma, 2.0 * tex)[None]
        inner = np.clip(S1, med - tol, med + tol)
        wf = np.clip(_gblur(Fp.astype(np.float32), 1.0) * 1.5, 0, 1)
        wf = np.maximum(wf, rem)
        out = x * (1 - wf[None]) + inner * wf[None]
        ex = out.mean(0) - med.mean(0)
        hot = ex > np.maximum(4 * sigma, 4 * tol[0])
        bump = ndimage.binary_dilation(hot, iterations=2) & ~ndimage.binary_opening(hot, iterations=3)
        out = np.where(bump[None], np.minimum(out, med + tol), out)
        return np.minimum(out, x).astype(np.float32)


    def noise_fill(x, S, sigma, seed=0):
        S = clean_residue(x, S, sigma)
        rem = (x - S).mean(0)
        sm = _gblur(rem, 1.0)
        where = (sm > 1.5 * sigma).astype(np.float32)
        if where.sum() < 4:
            return S
        reg = np.clip(_gblur(np.maximum(where, 0), 2.0) * 3, 0, 1)
        hp = S - _gblur(S, 1.0)
        loc = np.sqrt(np.maximum(_gblur(hp * hp, 8.0), 0)) * 1.75
        okw = 1 - np.clip(_gblur(where, 4.0) * 4, 0, 1)
        exp_sig = _gblur(loc * okw[None], 24.0) / np.maximum(_gblur(okw, 24.0), 1e-3)[None]
        add = np.sqrt(np.maximum(exp_sig ** 2 - loc ** 2, 0))
        sky = okw > 0.99
        d = hp.mean(0)
        rho = float(np.sum(d[:, :-1] * d[:, 1:] * sky[:, :-1]) / max(np.sum(d[:, :-1] ** 2 * sky[:, :-1]), 1e-12)) if sky.any() else 0.0
        rng = np.random.default_rng(seed)
        nz = rng.standard_normal(S.shape).astype(np.float32)
        if rho > 0.05:
            sb = math.sqrt(-1 / (4 * math.log(min(max(rho, 0.06), 0.95))))
            nz = _gblur(nz, sb); nz /= max(float(nz.std()), 1e-6)
        return np.minimum(S + reg[None] * add * nz, x).astype(np.float32)


    # ------------------------------------------------------------------------------------------------ star layer and Reduce Stars
    def star_layer(x, S, stretched):
        if stretched:
            return np.clip(1 - (1 - x) / np.maximum(1 - S, 1e-4), 0, 1).astype(np.float32)
        return (x - S).astype(np.float32)


    def combine(S, K, stretched):
        return (1 - (1 - S) * (1 - K)) if stretched else (S + K)


    def display_stretch(S):
        s = S[:, ::4, ::4].reshape(S.shape[0], -1)
        med = np.median(s, 1); mad = 1.4826 * np.median(np.abs(s - med[:, None]), 1)
        b = med - 2.8 * mad; vm = float(np.clip(np.mean((med - b) / (1 - b)), 1e-7, 0.2))
        m = (0.25 - 1) * vm / ((2 * 0.25 - 1) * vm - 0.25)
        return b.astype(np.float32)[:, None, None], float(m)


    def _mtf(v, m):
        return (m - 1) * v / ((2 * m - 1) * v - m)


    def _T(x, b, m):
        v = (x - b) / (1 - b)
        return np.where(v >= 0, _mtf(np.clip(v, 0, 1), m), v * (1 - m) / m)


    def _Ti(y, b, m):
        v = np.where(y >= 0, _mtf(np.clip(y, 0, 1), 1 - m), y * m / (1 - m))
        return b + (1 - b) * v


    def reduce_gain(L, r):
        a = 10.0 ** (-A_SPAN * r)
        g = a / (1 - (1 - a) * np.clip(L, 0, 1))
        if r > FADE0:
            t = (r - FADE0) / (1 - FADE0)
            g = g * (1 - t * t * (3 - 2 * t))
        return g


    def reduce_stars(S, K, r, stretched=False):
        if r <= 0:
            return combine(S, K, stretched)
        if r >= 1:
            return S.copy()
        if stretched:
            D = np.clip(K, 0, 1)
            return combine(S, D * reduce_gain(D.mean(0), r)[None], True).astype(np.float32)
        X = S + np.maximum(K, 0)
        b, mm = display_stretch(S)
        tS, tX = _T(S, b, mm), _T(X, b, mm)
        D = np.clip(1 - (1 - tX) / np.maximum(1 - tS, 1e-4), 0, 1)
        tO = 1 - (1 - tS) * (1 - D * reduce_gain(D.mean(0), r)[None])
        return np.minimum(np.maximum(_Ti(tO, b, mm), S), X).astype(np.float32)


    # ------------------------------------------------------------------------------------------------ engine
    class Engine:
        def __init__(self, model_path, session_factory):
            self.model_path = model_path
            self._run, self.provider, self.note = session_factory(model_path, probe_feed())

        def run(self, data, params=None, progress=None, cancel=None):
            p = dict(params or {})
            amount = float(min(max(p.get('amount', 1.0), 0.0), 1.0))
            mode = p.get('mode', 'auto')
            output = p.get('output', 'starless')
            if mode not in ('auto', 'linear', 'stretched'):
                raise ValueError("mode must be 'auto', 'linear' or 'stretched'")
            if output not in ('starless', 'stars'):
                raise ValueError("output must be 'starless' or 'stars'")
            data = np.asarray(data, np.float32)
            x, lay = to_chw(data)
            bad = ~np.isfinite(x)
            notes = []
            if bad.any():
                badu = bad.any(0)
                x = np.ascontiguousarray(_fill_nan(x, badu), np.float32)
                notes.append(f'{int(badu.sum())} NaN pixels kept as NaN')
            else:
                badu = None
                x = np.ascontiguousarray(x, np.float32)
            detected = bool(is_stretched(x))
            stretched = detected if mode == 'auto' else (mode == 'stretched')
            mode_used = 'stretched' if stretched else 'linear'
            notes.append(f'data: {mode_used}' + (' (auto)' if mode == 'auto' else ' (manual)'))
            H, W = x.shape[-2:]
            total = _n_tiles(H, W) + (_n_tiles(H // HALO_FACTOR, W // HALO_FACTOR) if min(H, W) // HALO_FACTOR >= 64 else 0)
            done = [0]

            def tick():
                if cancel is not None and cancel():
                    raise RuntimeError('Cancelled')
                done[0] += 1
                if progress is not None:
                    progress(min(done[0], total), total)
            if cancel is not None and cancel():
                raise RuntimeError('Cancelled')
            S0, sg = _starless(self._run, x, tick)
            S = noise_fill(x, S0, sg)
            K = star_layer(x, S, stretched)
            if output == 'stars':
                out = K
            elif amount <= 0:
                out = x.copy()
            else:
                out = reduce_stars(S, K, amount, stretched)
            out = np.ascontiguousarray(from_chw(out, lay), np.float32)
            stars = np.ascontiguousarray(from_chw(K, lay), np.float32)
            if badu is not None:
                bm = badu if lay in ('2d', '1hw', 'chw') else badu
                if lay == '2d':
                    out[bm] = np.nan; stars[bm] = np.nan
                elif lay in ('1hw', 'chw'):
                    out[:, bm] = np.nan; stars[:, bm] = np.nan
                else:
                    out[bm] = np.nan; stars[bm] = np.nan
            if progress is not None:
                progress(total, total)
            if self.note:
                notes.append(str(self.note))
            return out, {'notes': notes, 'stars': stars, 'mode_used': mode_used, 'sigma': float(sg)}
    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})


ENGINES['starless'] = _engine_starless()

# ============================================================================================ tools
def _tool_starless():
    # ---- Umbra Starless: tool layer ----
    REMOVAL_TIP = ('100% = all stars gone, 0% = stars untouched (the original image). In between every star gets smaller and dimmer '
                   'until it vanishes.')
    DATA_TIP = 'Linear or stretched. Auto detects it from the image; override if the guess is wrong.'
    OUTPUT_TIP = 'Stars only is the full star layer (ignores Star removal).'
    SAVE_TIP = 'Saves the star layer as <image>_stars.fit in the same folder (a new file; the image is not touched).'


    class StarlessPanel:
        def __init__(self, ui, layout, host):
            self.removal = ui.slider(layout, 'Star removal', 0, 100, 100, lambda v: f'{v}%', REMOVAL_TIP)
            self.data = ui.combo(layout, 'Data', [('Auto', 'auto'), ('Linear', 'linear'), ('Stretched', 'stretched')], DATA_TIP)
            self.out = ui.combo(layout, 'Output', [('Starless image', 'starless'), ('Stars only', 'stars')], OUTPUT_TIP)
            self.save = ui.check(layout, 'Also save the star layer as FITS next to the image', SAVE_TIP)

        def model_changed(self, path):
            pass

        def params(self):
            return {'amount': self.removal.value() / 100.0, 'mode': self.data.currentData(), 'output': self.out.currentData()}

        def settings(self):
            return dict(removal=self.removal.value(), data=self.data.currentData(), output=self.out.currentData(), save=self.save.isChecked())

        def set_settings(self, d):
            if isinstance(d.get('removal'), int):
                self.removal.setValue(d['removal'])
            for k, w in (('data', self.data), ('output', self.out)):
                i = w.findData(d.get(k))
                if i >= 0:
                    w.setCurrentIndex(i)
            if isinstance(d.get('save'), bool):
                self.save.setChecked(d['save'])


    class StarlessSpec(ToolSpec):
        def build_panel(self, ui, layout, host):
            return StarlessPanel(ui, layout, host)

        def add_cli(self, ap):
            ap.add_argument('--removal', type=float, default=1.0, help='starless: star removal 0..1 (1 = remove, lower = reduce stars; default 1)')
            ap.add_argument('--data', choices=['auto', 'linear', 'stretched'], default='auto', help='starless: data type (default auto)')
            ap.add_argument('--output', choices=['starless', 'stars'], default='starless', help='starless: write the starless image or the star layer')
            ap.add_argument('--stars-out', help='starless: also write the star layer to this FITS file (must not exist)')

        def check_cli(self, a, ap):
            if not 0 <= a.removal <= 1:
                ap.error('--removal must lie in [0, 1]')

        def params_from_args(self, a):
            return {'amount': a.removal, 'mode': a.data, 'output': a.output}

        def preview_params(self, p):
            return dict(p, output='starless')       # the preview shows the starless result; the star layer is one click away

        def label(self, p):
            return f"removal {float(p.get('amount', 1)):.2f} data {p.get('mode', 'auto')} output {p.get('output', 'starless')}"

        def post_apply(self, ctx, params, info):
            if not ctx.view.panel.save.isChecked() or info.get('stars') is None:
                return ''
            path = side_path(ctx.siril, 'stars')
            write_fits(path, info['stars'], siril_fits_header(ctx.siril), ctx.history + ' (star layer)')
            return f'Star layer saved to {path}.'

        def post_cli(self, a, dst, info, header, label):
            if not a.stars_out:
                return ''
            p = Path(a.stars_out).resolve()
            if p.exists() or p == dst:
                raise SystemExit(f'Output exists, refusing to overwrite: {p}')
            h = header.copy(); h['HISTORY'] = label + ' (star layer)'
            write_fits(p, info['stars'], h)
            return f'stars={p}'


    SPEC = StarlessSpec(
        key='starless', name='Umbra Starless', script='Umbra_Starless', repo=f'{GITHUB_USER}/umbra-starless',
        prefix='UMBRA_STARLESS_', folder='Umbra Starless',
        labels={'umbra_starless_v1.0': 'Umbra Starless v1.0 (default)', 'umbra_starless_v1.1_experimental': 'Umbra Starless v1.1 (experimental)'},
        default_stems=('umbra_starless_v1.0',),
        notes=['AI star remover for linear or stretched images.'], linear_only=False, alt=('Show star layer', 'stars'))
    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})


TOOLS['starless'] = _tool_starless().SPEC


if __name__ == '__main__':
    raise SystemExit(main([TOOLS[k] for k in ('starless',)], combined=False, doc=__doc__))

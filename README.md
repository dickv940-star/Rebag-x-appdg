# Rebag — Background Changer

A responsive, installable browser app for removing and replacing photo backgrounds.

## First release features
- Upload JPG, PNG, and WebP photos.
- AI background removal in the browser using Transformers.js and the ORMBG ONNX model.
- WebGPU acceleration where available, with a WASM fallback.
- Transparent, preset-color, custom-color, and uploaded-image backgrounds.
- Erase and restore brush for edge corrections.
- PNG, JPG (95%), and PDF export.
- PNG/JPG output keeps the source pixel dimensions. PDF uses one full-page image, preserves the source aspect ratio, and embeds the full-resolution canvas.
- PWA install support and cached application shell.

## Privacy and model
The uploaded photo is processed in the browser and is not sent to an application server. The first AI run downloads model files from Hugging Face; the browser may cache them for later use. A network connection is needed for the initial model download.

Dependencies:
- Transformers.js (Apache-2.0): https://github.com/huggingface/transformers.js
- ORMBG ONNX model (Apache-2.0 model lineage): https://huggingface.co/onnx-community/ormbg-ONNX

ORMBG is optimized for human subjects. Fine hair, transparent objects, busy backgrounds, and some product/object photos may need manual correction or a future model upgrade. Always review the cutout before exporting.

## Run locally
Serve this folder over HTTPS or localhost. Opening the HTML file directly from the filesystem will not work because the app uses ES modules and service workers.

## Notes
- PDF page dimensions are proportional to source pixels at a nominal 300 DPI. The embedded image remains at the original pixel dimensions; physical print size depends on PDF viewer/printer settings.
- JPG cannot preserve transparency. When the selected background is transparent, JPG export uses white.
- This first version is a static GitHub Pages app; it does not require a backend or API key.
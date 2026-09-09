# Image editing masks

Flux Fill requires a source image and a mask. Submit them to
`/v1/images/edits` as the `image` and `mask` fields (base64 PNG for JSON;
file parts for multipart).

The mask is a **paint selection**:

- Opaque grayscale or RGB/RGBA/LA PNG: white selects editing, black keeps.
- PNG with transparency: alpha defines the painted selection; opaque pixels
  select editing and transparent pixels keep. Hidden RGB is ignored.
- A fully opaque alpha channel is not a full-image selection by itself.
- Empty selections are rejected before inference.

This is not an inverted-alpha convention where transparency means erase.
Convert masks using that convention to an opaque white/black selection first.
The app's mask painter exports this paint-selection format.

The adapter resizes the source and mask to the requested output dimensions.
Generative output is not a pixel-exact preservation guarantee outside the mask.

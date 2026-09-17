fits2image
==========

Convert astronomical FITS files into JPEG, TIFF or in-memory Pillow images.
The library scales pixel brightness for display, creates color composites,
and uses image headers to orient previews toward north up and east left.

The fits2image package provides common libraries for thumbnail-to-archive-service,
ptr_datalab, neoexchange and serol. It is also a dependency of the TOM Toolkit
(tomtoolkit), through which most third-party users pull it.

Installation
------------

Requires Python 3.10 or newer. Install with:

```bash
pip install fits2image
```

NumPy, Astropy and Pillow are installed as dependencies. Import functions from
fits2image.conversions, fits2image.scaling or fits2image.orientation, as shown
in the examples below.

Quick start
-----------

Create a JPEG preview that fits within 800 by 800 pixels, or a TIFF preview
that fits within 1024 by 1024 pixels:

```python
from fits2image.conversions import fits_to_jpg, fits_to_tif

success = fits_to_jpg("frame.fits", "./preview.jpg", width=800, height=800)
if not success:
    raise RuntimeError("Could not create preview")

fits_to_tif("frame.fits", "./preview.tif", width=1024, height=1024)
```

Use an output path with a directory component, such as "./preview.jpg" or
"previews/frame.jpg". Missing directories are created automatically; a bare
filename currently fails when the writer tries to create an empty directory path.

Choose the function that fits your task:

- fits_to_jpg or fits_to_tif: save a preview of a FITS image.
- fits_to_img: choose another output format or use the full set of options.
- fits_to_zoom_slice_jpg: save a square tile from part of an image.
- multi_fits_to_img: assign colors to images and blend them into one composite.
- get_scaled_image: get a Pillow image to resize, edit or save yourself.

File conversion API (fits2image.conversions)
--------------------------------------------

### Converting an image: fits_to_img

Convert a FITS image to a Pillow-supported format, typically 'jpeg' or 'TIFF'.

```python
fits_to_img(path_to_fits, path_to_output, file_type, width=200, height=200,
            progressive=False, label_text='', label_font='DejaVuSansMono.ttf',
            zmin=None, zmax=None, gamma_adjust=2.5, contrast=0.1, quality=95,
            color=False, percentile=99.5, median=False)
```

path_to_fits accepts a string path, pathlib.Path, or an object with open('rb').
It can also be a list of such inputs. FITS compression is handled by Astropy.

Size and brightness:

- width and height are maximum output dimensions in pixels. The image keeps
  its proportions after orientation and is never enlarged.
- zmin and zmax set the darkest and brightest input values to display. Values
  outside these limits are clipped. Supply both limits for manual control, or
  leave both as None to estimate them from a sample of the pixels.
- contrast controls the automatic brightness fit (zscale). Automatic scaling
  uses the sample median as its lower limit and the fitted upper limit.
- gamma_adjust controls gamma correction; use 1 to leave the linear brightness
  scale unchanged.
- median=True subtracts the median brightness from the scaled image, then
  stretches the remaining values using percentile as the upper reference.
  This adjusts the background brightness; it does not smooth nearby pixels.
  percentile is ignored when median=False.

Multiple inputs and color:

- color=True requires exactly three inputs in red, green, blue order.
  Each channel is scaled separately and then combined. If sizes differ, the
  library takes a centered crop from each image so they all share the smallest
  width and height. Corresponding stars must already line up across the images;
  the library does not align them.
- With color=False, each input produces a separate grayscale image saved in
  RGB mode. For a list, subsequent filenames append '-1', '-2', etc. to the
  entire output path (for example, preview.jpg-1).
- For multiple inputs, supply zmin and zmax as lists with one entry per image.
  Use None entries to select automatic scaling for individual images.

Labels and output settings:

- label_text adds a label near the lower left after resizing. label_font names
  a TrueType font available to Pillow. If the label cannot be drawn, conversion
  continues with a warning.
- quality controls JPEG compression quality. progressive=True writes a
  progressive JPEG, which can display in stages as it downloads. These options
  are passed to Pillow and may have no effect for other output formats.

For a single input, manual zmin and zmax can be scalars or one-element lists.
Supply both together: a scalar zmin with zmax=None currently raises TypeError.
If either limit is None after normalization, the image uses automatic scaling.

Returns True after saving, or False for handled failures such as missing input
files, mismatched limit-list lengths, an incorrect color-channel count, or output
I/O errors. Other errors, including malformed FITS data and unsupported options,
can raise exceptions rather than returning False.

### JPEG and TIFF shortcuts

```python
fits_to_jpg(path_to_fits, path_to_jpg, width=200, height=200,
            progressive=False, label_text='', label_font='DejaVuSansMono.ttf',
            zmin=None, zmax=None, gamma_adjust=2.5, contrast=0.1, quality=95,
            color=False, percentile=99.5, median=False)
```

fits_to_jpg calls fits_to_img with file_type='jpeg'. Its options and return
value work the same way.

```python
fits_to_tif(path_to_fits, path_to_tif, width=200, height=200, contrast=0.1,
            gamma_adjust=2.5, quality=100, percentile=99.5, median=False,
            progressive=False)
```

fits_to_tif calls fits_to_img with file_type='TIFF'. It accepts only the options
shown above; use fits_to_img for TIFF labels, manual limits, or color. The TIFF
contains an 8-bit display image, so retain the FITS file for the original data.

### Creating an RGB composite

Example RGB composite with explicit per-channel limits:

```python
from fits2image.conversions import fits_to_jpg

success = fits_to_jpg(
    ["red.fits", "green.fits", "blue.fits"], "./rgb.jpg",
    color=True, width=1000, height=1000,
    zmin=[100, 120, 90], zmax=[2000, 2400, 1800], gamma_adjust=2.5,
)
```

### Saving a square tile: fits_to_zoom_slice_jpg

```python
fits_to_zoom_slice_jpg(path_to_fits, path_to_jpg, row=0, col=0, side=200,
                       zlevel=0, zfactor=1.25, progressive=False,
                       label_text='', label_font='DejaVuSansMono.ttf',
                       zmin=None, zmax=None, gamma_adjust=2.5, contrast=0.1,
                       quality=75)
```

Write one square JPEG tile from a FITS file. The image is oriented first, then
row and col select a tile starting from the upper left. Both indices start at 0.

The function crops a square with an edge length of
step = int(side / zfactor**zlevel) source pixels. Its upper-left corner is at
x = col * step and y = row * step. The crop is then reduced, if necessary, so
its width and height are each at most side pixels. Smaller crops are not enlarged.

For example, side=200 and zlevel=0 select a 200 by 200 pixel crop. With the
default zfactor=1.25, zlevel=1 selects a 160 by 160 pixel crop and saves a
160 by 160 pixel JPEG. Choose parameters that keep step at least 1.

Returns True on success or False for a missing file or output I/O failure;
other errors can raise exceptions. Label and scaling options work as above.

### Blending images with custom colors: multi_fits_to_img

```python
multi_fits_to_img(input_fits, path_to_output, blending_algorithm='sum',
                  width=200, height=200, file_type='jpeg',
                  progressive=False, quality=95)
```

Blend a nonempty list of input dictionaries into one RGB image. Each dictionary
must contain:

- fits_path: a FITS input as above, or fits_data: a writable 2D floating-point
  NumPy array. fits_path takes precedence if both are supplied.
- scale_algorithm: 'zscale' (the only supported value).
- zmin and zmax: numeric clipping limits. Despite the algorithm name, these
  must be explicit; None does not request automatic estimation here.
- color: three weights (red, green, blue), each between 0 and 1.

blending_algorithm accepts 'sum', 'min', 'max', 'multiply', 'screen', or 'overlay'.
file_type must be lowercase 'jpeg' or 'tiff'. Inputs are oriented together and
center-cropped to a common size before blending, then a fixed gamma of 2.5 is
applied and the result is resized to fit within width and height.

This function modifies its input dictionaries and pixel arrays. Pass array
copies if you need to retain the original data, as in the example below. Arrays
must support floating-point arithmetic, including those read from FITS files;
integer data can raise NumPy casting errors. The dictionaries gain a
scaled_image entry containing the working array.

Returns True on save, False for output I/O errors, and raises ValueError for
invalid input fields or unsupported algorithms or formats. Read and scaling
errors can also raise exceptions.

```python
import numpy as np
from fits2image.conversions import multi_fits_to_img

data = np.linspace(0, 1000, 256 * 256).reshape(256, 256)
success = multi_fits_to_img(
    [{"fits_data": data.copy(), "scale_algorithm": "zscale",
      "zmin": 0, "zmax": 1000, "color": (1.0, 0.5, 0.0)}],
    "./orange.jpg", width=256, height=256,
)
```

In-memory images and scaling (fits2image.scaling)
-------------------------------------------------

### Getting a Pillow image: get_scaled_image

```python
get_scaled_image(path_to_fits, zmin=None, zmax=None, contrast=0.1,
                 gamma_adjust=2.5, flip_v=True, percentile=99.5,
                 median=False, transform=DERIVE_FROM_HEADER)
```

Return an oriented, 8-bit grayscale Pillow Image at the original resolution.
You can then resize, edit or save it with Pillow. Scaling and median options
behave as in fits_to_img. This function accepts scalar limits directly and
automatically scales if either is None. Errors raise exceptions. See the
Orientation section for transform and flip_v.

```python
from fits2image.scaling import get_scaled_image

image = get_scaled_image("frame.fits", zmin=100, zmax=2000)
image.thumbnail((800, 800))
image.save("preview.png")
```

### Reading FITS data and headers

These readers select the first HDU containing a nonempty 2D image:

- get_reduced_dimensionality_data(path_to_frame) returns (data, header): a 2D
  NumPy array and an Astropy Header. It selects the first nonempty 2D image HDU,
  skipping tables, empty images and cubes. It does not combine extensions.
  Missing primary-header keywords are merged into the selected HDU's header,
  with the selected HDU taking precedence. The merged header is for lookups,
  not for writing back to FITS. If no eligible image is found, the function
  raises an exception.
- get_frame_header(path_to_frame) returns that same merged header without
  reading the pixel array.

### Working with arrays

These helpers scale brightness without changing orientation or resizing:

- auto_scale(path_to_frame, nsamples=2000, max_val=255, contrast=0.1,
  gamma_adjust=2.5, max_fit_iterations=1) returns a scaled uint8 NumPy array.
  auto_scale_data(data, header, nsamples=2000, max_val=255, contrast=0.1,
  gamma_adjust=2.5, max_fit_iterations=1) does the same with already-read data.
  Sampling requires at least nsamples pixels; reduce nsamples for small images,
  or supply explicit limits to get_scaled_image.
- linear_scale(data, zmin, zmax, max_val=255, gamma_adjust=2.5) clips, scales,
  and gamma-corrects into a new uint8 array. Equal limits are expanded by one
  on either side with a warning. Keep max_val within the 8-bit range.
- quick_scale_image(input_fits) uses the input dictionary described for
  multi_fits_to_img, returning scaled floating-point data without gamma or
  orientation. It modifies data in place and, for file inputs, adds bitpix,
  saturate and transform metadata to the dictionary. color is not used here.
- percentile_scale(path_to_frame, lower_percentile=5.0, upper_percentile=99.0)
  reads via astropy.io.fits.getdata and returns an 8-bit percentile stretch
  without gamma correction. It does not use the merged-header reader above.
- recalculate_median(data, percentile=99.5) returns a uint8 array after median
  background subtraction and percentile stretching.

To combine three already-scaled grayscale Pillow images, use
stack_images(images_to_stack). It takes the images in red, green, blue order
and returns a color Pillow Image, using centered crops if their sizes differ.

Orientation (fits2image.orientation)
------------------------------------

Since 1.0.0, conversions and get_scaled_image orient images toward north up and
east left using CD1_1, CD1_2, CD2_1 and CD2_2. Rotation is snapped to the nearest
90 degrees, not an exact sky reprojection. A quarter turn swaps width and height
before thumbnailing or tiling. Without a usable CD matrix the default is a
vertical flip; alternative WCS encodings such as PC/CDELT are not interpreted.

Color stacks and multi_fits_to_img use the first usable orientation for every
input, warning if the headers imply different orientations. With no usable transform, all
inputs receive the vertical flip. Raw fits_data inputs have no header; an
optional transform entry can supply their orientation explicitly.

### Orientation helpers

- get_cd_matrix(header) returns a 2x2 NumPy matrix or None for missing,
  invalid CD values (including nonfinite values or a singular matrix).
- orientation_transform(header) returns (mirror, k), or None when unavailable.
  mirror is a left-right reflection applied first; k is the number of subsequent
  counterclockwise quarter turns (0 through 3).
- apply_orientation(image, transform) and apply_orientation_array(array,
  transform) apply an explicit (mirror, k) to a Pillow image or 2D NumPy array.
- orient_image(image, header, flip_v=True, frame='',
  transform=DERIVE_FROM_HEADER) derives the transform from the header by default.
  frame identifies the input in fallback warnings. An explicit tuple overrides
  the header; transform=None forces the flip_v fallback. flip_v=False alone
  does not disable a usable header-derived orientation.
- orient_array(array, transform, flip_v=True) applies a resolved transform,
  falling back to a vertical flip when transform=None and flip_v=True.

To preserve the stored pixel orientation when obtaining a Pillow image:

```python
image = get_scaled_image("frame.fits", transform=None, flip_v=False)
```

Release history
---------------

- v 0.1.0 - Initial version with conversion code from frame_database
- v 0.1.1 - Changed the fits_to_* functions to return False if they fail (and log warnings), otherwise return True
  - Added a quality parameter for jpeg conversion quality for fits_to_* functions
- v 0.1.2 - Changed behaviour to automatically use the first quadrant of a data cube, instead of crashing
- v 0.1.3 - If zmax/zmin calculated to be the same, give them a slit offset to avoid dividing by zero
- v 0.1.4 - Check if path exists and create it if not before saving file with pillow.
- v 0.1.5 - Fix problem creating paths
- v 0.2.0 - Add color image support. fits_2_image now accepts list of images, and if will stack them if the list
  is of length 3 and the color=True parameter is set. Expects order to be RVB.
- v 0.2.1 - Remove cosmic ray removal which slows down image generation considerably and misrepresents the actual data
  - Remove median filter which slows down image generation
- v 0.3.0 - Add median filter support
- v 0.3.1 - Add support for multi extension fits files
- v 0.3.2 - Fix bug in multi extension fits handling
- v 0.4.0 - Use astropy instead of cfitsio
- v 0.4.2 - Remove dependency on filesystem paths
- v 0.4.3 - Fix memory leak
- v 0.4.4 - Fix for images where the zeroth HDU has shape (0,0)
- v 0.4.5 - Add github actions to push to PyPI
- v 0.4.6 - No functional change
- v 0.4.7 - No functional change
- v 0.4.8 - Add fits_to_tif and fits_to_jpg convenience functions
- v 0.4.9 - Allow fits_to_img to take a list of zmin and zmax parameters (one for each color channel) when creating an RGB stack.
- v 0.4.10 - Crop fallback whens stacking images that can't be aligned and have different dimensions
- v 0.4.11 - Allow a gamma adjustment to be passed through to the color stacking path
- v 1.0.0 - Output images are oriented north up and east left from the frame's CD matrix, snapped
  to the nearest 90 degrees. This changes the output of all callers so bumping a major version.
  A frame with no WCS falls back to a fixed vertical flip.
  A quarter turn swaps the width and height of a non-square frame,
  so the output comes out in the other aspect.

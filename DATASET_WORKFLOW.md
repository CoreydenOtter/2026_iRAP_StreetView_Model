# Dataset Collection and Annotation Workflow

This document describes the image collection and annotation utilities
used in the **Smart Road Safety Assessment Using Multimodal AI and
Street-View Imagery** project.

The repository utilities support a pipeline for:

1.  sampling road locations from OpenStreetMap;
2.  retrieving suitable Google Street View imagery;
3.  automatically selecting a road-centred camera heading;
4.  uploading candidate images to Google Drive;
5.  manually encoding road-safety attributes using a schema-driven
    annotation interface; and
6.  storing annotations in a structured JSON dataset for later model
    training and evaluation.

> **Important:** API keys, OAuth credentials, Google Drive folder IDs,
> generated access tokens, and private datasets should not be committed
> to GitHub.

------------------------------------------------------------------------

## Repository Components

### `Image_Retrival.py`

Collects candidate road images for the dataset.

The script:

-   samples drivable roads from OpenStreetMap through the Overpass API;
-   currently defines collection regions around Brisbane and the Gold
    Coast;
-   selects random points along returned road geometries;
-   checks whether outdoor Google Street View imagery is available near
    each point;
-   retrieves views at headings of `0`, `90`, `180`, and `270` degrees;
-   uses an OpenCV heuristic to score how well each image is centred on
    the road;
-   keeps the highest-scoring heading; and
-   uploads the selected JPEG directly to a configured Google Drive
    folder.

The road-centering heuristic uses edge detection and Hough line
detection in the lower part of the image. It rewards useful
non-horizontal road-line structure and convergence toward the image
centre, while applying penalties for views dominated by blue sky/water
or sand/beige regions.

Generated filenames contain the sequence number, collection region,
panorama coordinates, and selected heading.

Example:

``` text
0001_brisbane_-27.47000_153.02000_h90.jpg
```

### `Encoder.py`

Provides a Tkinter graphical interface for manually encoding images
stored in Google Drive.

The encoder:

-   authenticates to Google Drive using OAuth;
-   reads annotation fields from `Attributes_Schema.json`;
-   dynamically creates controls for Boolean and single-choice
    attributes;
-   downloads the current image from the source Drive folder for
    display;
-   supports keyboard-oriented annotation;
-   stores annotations in `annotations.json`;
-   uploads/updates the annotation file in Google Drive;
-   moves successfully encoded images from the source folder to the
    encoded folder; and
-   allows unsuitable images to be moved to Google Drive Trash.

Images are selected randomly rather than strictly sequentially. This
reduces the likelihood of multiple encoder instances repeatedly
selecting the same next image.

### `Attributes_Schema.json`

Defines the annotation structure used by the encoder.

The schema is named `irap_image_annotation_schema`, version `1.0`, and
treats the annotation unit as a single image or image sequence
representing a road segment.

It separates attributes according to whether they can be determined
directly or partially from imagery. Fields that are not reliably
image-only can remain optional or `null`.

The schema contains road-safety attributes across categories including:

-   observed road-user flow;
-   speed;
-   mid-block road characteristics;
-   roadside characteristics;
-   intersections;
-   land use; and
-   pedestrian, bicycle, motorcycle and school-zone facilities.

Because the encoder builds its controls from this schema, changes to
supported attributes can generally be made in the JSON schema rather
than by manually rebuilding the GUI.

------------------------------------------------------------------------

## Processing Pipeline

``` text
OpenStreetMap / Overpass
          |
          v
 Random road location
          |
          v
Street View metadata check
          |
          v
Retrieve four headings
          |
          v
OpenCV road-centering score
          |
          v
Select best candidate image
          |
          v
Upload to Google Drive
          |
          v
Schema-driven manual encoding
          |
          +----------------------+
          |                      |
          v                      v
 Save annotation          Reject unsuitable image
          |                      |
          v                      v
Move to encoded folder      Drive Trash
          |
          v
    annotations.json
          |
          v
Model training / validation / testing
```

------------------------------------------------------------------------

## Requirements

The scripts use Python 3 and the following main packages:

``` text
opencv-python
numpy
requests
Pillow
google-auth
google-auth-oauthlib
google-api-python-client
```

Tkinter is also required for the annotation GUI. Depending on the
operating system and Python installation, Tkinter may already be
included or may need to be installed separately.

A typical installation is:

``` bash
pip install opencv-python numpy requests Pillow google-auth google-auth-oauthlib google-api-python-client
```

------------------------------------------------------------------------

## Google API Configuration

### Street View

`Image_Retrival.py` requires a Google Maps Platform API key with access
to the required Street View service.

Set the key locally:

``` python
API_KEY = "YOUR_API_KEY"
```

Do **not** commit the real key.

### Google Drive OAuth

Both utilities that access Drive require a Google OAuth desktop client
configuration.

Place the downloaded OAuth client file beside the scripts as:

``` text
credentials.json
```

On the first successful authentication, the scripts create:

``` text
token.json
```

Both files should be excluded from version control.

### Drive folders

Configure the appropriate folder IDs locally.

For retrieval:

``` python
DRIVE_FOLDER_ID = ""
```

For encoding:

``` python
SOURCE_FOLDER_ID = ""
ENCODED_FOLDER_ID = ""
```

The source folder contains images waiting to be encoded. After an
annotation is saved, the encoder moves the image into the encoded
folder.

------------------------------------------------------------------------

## Recommended `.gitignore`

At minimum, the repository should exclude credentials, tokens,
environments, generated annotations, and local datasets:

``` gitignore
# Google credentials
credentials.json
token.json

# Secrets / environment files
.env
.env.*

# Python
__pycache__/
*.py[cod]
.venv/
venv/

# IDE / OS
.vscode/
.idea/
.DS_Store

# Generated/private dataset content
annotations.json
annotations_cleaned.json
*.jpg
*.jpeg
*.png
dataset/
datasets/
training_data/
Augmented_Dataset/

# Model outputs
checkpoints/
outputs/
predictions/
*.pt
*.pth
*.safetensors
```

If example images or small public dataset samples are intentionally
included, add exceptions for those specific paths.

------------------------------------------------------------------------

## Running Image Retrieval

Before running:

1.  configure the Street View API key;
2.  configure the target Google Drive folder ID;
3.  place `credentials.json` beside the script; and
4.  install the required Python dependencies.

Run:

``` bash
python Image_Retrival.py
```

The script asks:

``` text
How many images do you want?
```

It then continues sampling candidate roads until the requested number of
images has been uploaded or the maximum number of attempts is reached.

------------------------------------------------------------------------

## Running the Encoder

Before running:

1.  place `Attributes_Schema.json` beside the encoder;
2.  configure `SOURCE_FOLDER_ID`;
3.  configure `ENCODED_FOLDER_ID`;
4.  place `credentials.json` beside the script; and
5.  install Pillow and the Google API dependencies.

Run:

``` bash
python Encoder.py
```

### Keyboard controls

  Shortcut                Action
  ----------------------- -----------------------------------------
  `Ctrl + S`              Save annotation and move image
  `Ctrl + Right`          Load another random image
  `Ctrl + Left`           Load another random image
  `Ctrl + D`              Delete/reject the current image
  `F5`                    Refresh images from Drive
  `Up / Down`             Move between annotation fields
  `Left / Right`          Cycle choices in a selected drop-down
  `Space`                 Cycle forward through drop-down choices
  `Page Up / Page Down`   Scroll through the form
  `Home / End`            Move to top/bottom of the form

------------------------------------------------------------------------

## Annotation Output

Annotations are stored in a JSON structure containing the schema
information and an `annotations` mapping.

Each image record includes information such as:

``` json
{
  "drive_file_id": "...",
  "image_id": "...",
  "image_name": "...",
  "country": null,
  "source_folder_id": "...",
  "encoded_folder_id": "...",
  "attributes": {
    "speed_sign_visible": "true",
    "number_of_lanes": "two_lanes",
    "road_condition": "good"
  }
}
```

Unselected attributes are stored as `null`.

The exact available attributes and choices are controlled by
`Attributes_Schema.json`.

------------------------------------------------------------------------

## Schema Design

The current schema is intended for image-based road-safety assessment.
Examples of supported attributes include:

-   visible speed signs and speed-limit classes;
-   lane count and road curvature;
-   median type;
-   skid resistance and road condition;
-   roadworks and delineation;
-   street lighting;
-   centre-line and shoulder rumble strips;
-   roadside objects and paved shoulders;
-   intersection type and channelisation;
-   land use and area type;
-   pedestrian crossing facilities and fencing;
-   bicycle facilities; and
-   school-zone warnings.

Some attributes are inherently easier to infer from a single road image
than others. The schema therefore records image observability and allows
uncertain or unavailable values to remain unset.

------------------------------------------------------------------------

## Reproducibility Notes

The collection process contains random sampling at several stages:

-   selection of collection region;
-   selection of an OpenStreetMap road;
-   interpolation of a point along the selected road; and
-   random image selection in the annotation interface.

As currently implemented, no fixed random seed is applied. Therefore,
running the collection script again is not expected to produce the same
sequence of road locations or images.

For experiments requiring exact reproducibility, consider recording:

-   the image filename;
-   panorama coordinates;
-   selected heading;
-   collection date;
-   Street View panorama identifier, where available;
-   schema version; and
-   dataset split assignment.

------------------------------------------------------------------------

## Data and Privacy Considerations

Google Street View imagery is retrieved through Google services rather
than distributed by this repository. Users of this code are responsible
for ensuring that their use of imagery, APIs, stored data, and derived
datasets complies with the relevant service terms, institutional
requirements, and research ethics obligations.

For a public GitHub repository, it is generally preferable to publish
the **code, schema, documentation, and non-sensitive metadata**
separately from private credentials and large image datasets.

------------------------------------------------------------------------

## Suggested Repository Layout

``` text
smart-road-safety-dataset/
├── README.md
├── DATASET_WORKFLOW.md
├── Attributes_Schema.json
├── Encoder.py
├── Image_Retrival.py
├── requirements.txt
├── .gitignore
└── examples/
    └── annotation_example.json
```

For consistency, consider renaming `Image_Retrival.py` to
`Image_Retrieval.py` before publishing. If renamed, update any
documentation or scripts that reference the old filename.

------------------------------------------------------------------------

## Research Context

These utilities form the data preparation stage of a wider research
workflow investigating multimodal vision-language models for automated
road-safety attribute classification from street-level imagery.

The collection and encoding tools are intentionally separated from model
training. This keeps dataset creation, annotation, model development,
and evaluation as distinct stages and makes it easier to revise the
schema or dataset without coupling those changes to a particular model
implementation.

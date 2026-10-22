# Smart Road Safety Assessment Using Multimodal AI and Street-View Imagery

A research project investigating the use of **multimodal vision-language
models (VLMs)** to automate road-safety attribute classification from
street-level imagery.

The project develops a pipeline for collecting Google Street View
imagery, manually encoding road attributes using an iRAP-inspired
annotation schema, and preparing the resulting dataset for multimodal AI
training and evaluation.

## Project Overview

Road-safety assessment frameworks require many characteristics of a road
environment to be identified and encoded. Manually performing this
process across large road networks can require significant time and
specialist effort.

This project investigates whether multimodal AI models can identify
road-safety attributes directly from street-level imagery.

The current workflow consists of:

``` text
OpenStreetMap road sampling
        ↓
Google Street View retrieval
        ↓
Automatic heading selection
        ↓
Google Drive image storage
        ↓
Manual schema-based annotation
        ↓
Structured annotation dataset
        ↓
Multimodal model training and evaluation
```

The repository currently focuses on the **image collection and dataset
annotation components** of this workflow.

------------------------------------------------------------------------

## Repository Contents

``` text
.
├── README.md
├── DATASET_WORKFLOW.md
├── Attributes_Schema.json
├── Encoder.py
└── Image_Retrieval.py
```

### `Image_Retrieval.py`

Collects candidate road images for the dataset.

The script:

-   retrieves drivable road geometry from OpenStreetMap using the
    Overpass API;
-   samples random road locations;
-   checks for available outdoor Google Street View imagery;
-   retrieves candidate views at four headings;
-   uses OpenCV-based image analysis to estimate which view is best
    centred on the road; and
-   uploads the selected image to Google Drive.

The current collection regions are centred around **Brisbane and the
Gold Coast, Queensland, Australia**.

> The original script may be named `Image_Retrival.py`. Renaming it to
> `Image_Retrieval.py` is recommended before publishing the repository.

### `Encoder.py`

Provides a graphical interface for manually annotating the collected
images.

The encoder:

-   loads images from Google Drive;
-   dynamically builds annotation controls from
    `Attributes_Schema.json`;
-   supports keyboard-based annotation;
-   stores annotations in JSON format;
-   uploads updated annotations to Google Drive; and
-   moves completed images into a separate encoded-image folder.

### `Attributes_Schema.json`

Defines the road attributes and valid annotation choices used by the
encoder.

The schema is designed around attributes that can be directly or
partially determined from street-level imagery.

Example attribute groups include:

-   observed road users;
-   speed-related features;
-   lane and road geometry;
-   road condition;
-   medians;
-   roadworks;
-   delineation;
-   street lighting;
-   roadside objects;
-   paved shoulders;
-   rumble strips;
-   intersections;
-   surrounding land use;
-   pedestrian facilities;
-   bicycle facilities; and
-   school-zone features.

### `DATASET_WORKFLOW.md`

Contains detailed documentation for the image collection and annotation
pipeline, including configuration, keyboard controls, annotation
structure, reproducibility considerations, and recommended repository
security practices.

------------------------------------------------------------------------

## Model Training and Evaluation

The [`Models/`](Models/) directory contains training, testing and evaluation scripts for Qwen2.5-VL, Janus-Pro and a VGG16 baseline. See **[Models/README.md](Models/README.md)** for setup, commands and experiment details.

---

## Annotation Schema

The annotation schema is:

``` text
irap_image_annotation_schema
Version: 1.0
```

The annotation unit is a **single image or image sequence representing a
road segment**.

Attributes contain information describing whether they can be determined
from imagery. This allows the dataset to distinguish between directly
observable road characteristics and attributes that may require
additional information.

A simplified annotation record resembles:

``` json
{
  "image_id": "example_image",
  "attributes": {
    "speed_sign_visible": "true",
    "number_of_lanes": "two_lanes",
    "curvature": "straight",
    "road_condition": "good",
    "area_type": "urban"
  }
}
```

The complete set of fields and permitted values is defined in
`Attributes_Schema.json`.

------------------------------------------------------------------------

## Requirements

The utilities are written in Python and use packages including:

``` text
opencv-python
numpy
requests
Pillow
google-auth
google-auth-oauthlib
google-api-python-client
```

Install the main dependencies with:

``` bash
pip install opencv-python numpy requests Pillow google-auth google-auth-oauthlib google-api-python-client
```

`Encoder.py` also requires **Tkinter**. Tkinter is included with many
Python installations but may need to be installed separately depending
on the operating system.

------------------------------------------------------------------------

## Google API Setup

The utilities use Google Street View and Google Drive.

### Street View API

Configure a Google Maps Platform API key in the image retrieval script:

``` python
API_KEY = "YOUR_API_KEY"
```

The API key should be kept private and should **never be committed to
the repository**.

### Google Drive

Google Drive access uses OAuth.

Download a Google OAuth desktop client configuration and save it locally
as:

``` text
credentials.json
```

After the first successful authentication, a token file is generated:

``` text
token.json
```

Both files must remain private.

The relevant Drive folder IDs must also be configured locally in the
scripts.

------------------------------------------------------------------------

## Collecting Images

Configure the API key and Google Drive destination folder before running
the retrieval utility.

Run:

``` bash
python Image_Retrival.py
```

The program asks how many images should be collected.

For each successful sample, it:

1.  selects a road location;
2.  verifies Street View availability;
3.  retrieves four candidate headings;
4.  evaluates each candidate;
5.  selects the highest-scoring road view; and
6.  uploads the image to Google Drive.

------------------------------------------------------------------------

## Annotating Images

Ensure the following files are available:

``` text
Encoder.py
Attributes_Schema.json
credentials.json
```

Configure the source and encoded Google Drive folder IDs and then run:

``` bash
python Encoder.py
```

The annotation interface displays an image alongside the fields
generated from the schema.

Useful shortcuts include:

  Shortcut                Action
  ----------------------- -----------------------------
  `Ctrl + S`              Save annotation
  `Ctrl + Right`          Load another image
  `Ctrl + Left`           Load another image
  `Ctrl + D`              Reject/delete current image
  `F5`                    Refresh Drive images
  `Up / Down`             Change selected attribute
  `Left / Right`          Change selected value
  `Space`                 Cycle through values
  `Page Up / Page Down`   Scroll annotation form
  `Home / End`            Move to top/bottom

After a successful annotation, the image is moved from the source Drive
folder to the encoded folder.

------------------------------------------------------------------------

## Dataset Workflow

More detailed information about dataset creation is available in:

**[DATASET_WORKFLOW.md](DATASET_WORKFLOW.md)**

This includes information about:

-   OpenStreetMap sampling;
-   Street View retrieval;
-   automatic road-view selection;
-   Google Drive integration;
-   schema-driven annotation;
-   annotation output;
-   reproducibility; and
-   dataset/privacy considerations.

------------------------------------------------------------------------

## Security

Before publishing this repository, ensure that the following are **not
committed**:

``` text
credentials.json
token.json
.env
API keys
Google Drive folder IDs
private image datasets
generated annotation datasets containing restricted data
model checkpoints
```

A `.gitignore` should be used to prevent these files from accidentally
being uploaded.

If a credential has previously been committed to Git, removing it from
the latest commit alone may not be sufficient. The credential should be
revoked or rotated.

------------------------------------------------------------------------

## Research Scope

The wider research project uses the generated dataset to investigate
multimodal AI for road-safety assessment.

The overall experimental workflow is:

``` text
Image Collection
      ↓
Manual Annotation
      ↓
Dataset Cleaning
      ↓
Train / Validation / Test Split
      ↓
Zero-Shot Evaluation
      ↓
Model Fine-Tuning
      ↓
Fine-Tuned Evaluation
      ↓
Attribute-Level Comparison
```

Keeping image collection, annotation, and model training as separate
stages allows the dataset and annotation schema to be revised
independently of the selected multimodal model.

------------------------------------------------------------------------

## Limitations

The current data collection approach has several limitations:

-   road locations are randomly sampled rather than systematically
    covering the complete road network;
-   Street View availability varies between locations;
-   only a small set of headings is evaluated for each panorama;
-   automatic road-centering is heuristic rather than semantic;
-   some road-safety attributes cannot be reliably inferred from a
    single image; and
-   manual annotation remains necessary for creation of the ground-truth
    dataset.

These limitations are important when interpreting subsequent model
performance.

------------------------------------------------------------------------

## Documentation

For implementation details, see:

-   **[Dataset Collection and Annotation
    Workflow](DATASET_WORKFLOW.md)**
-   **[Annotation Schema](Attributes_Schema.json)**

------------------------------------------------------------------------

## Project Status

This repository forms part of an ongoing university research project.

Current development includes:

-   Street View dataset collection;
-   schema-based manual annotation;
-   dataset preparation;
-   multimodal zero-shot evaluation;
-   multimodal model fine-tuning; and
-   comparison of model performance across road-safety attributes.

------------------------------------------------------------------------

## Disclaimer

This repository is intended for **academic and research purposes**.

Google Street View imagery and Google APIs remain subject to Google's
applicable terms and policies. OpenStreetMap data is subject to its
applicable licence and attribution requirements. Users are responsible
for ensuring that their use of imagery, APIs, and derived datasets
complies with the relevant terms and institutional requirements.

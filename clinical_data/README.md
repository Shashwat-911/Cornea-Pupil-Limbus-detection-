# Clinical Data

This directory contains ophthalmic surgical images and their annotations, used
for training and evaluating the pupil/limbus detector.

## Contents

- `clean/` — Curated surgical eye images with suction ring
- `annotations/` — Segmentation masks and geometric annotations
- `training_data/` — Training image/mask pairs

## Distribution

**This data is NOT included in the public repository.**

The clinical images are real ophthalmic surgical photographs. They are subject
to:

- Patient privacy and data protection regulations (HIPAA, GDPR, etc.)
- Institutional ethics board requirements
- Specific data use agreements

To use or evaluate this system you need **separately authorized,
de-identified** ophthalmic image data. Contact the project owner for access
arrangements.

## Do not upload clinical data

Do not commit clinical images, masks, or annotations to any public Git
repository. Always verify that any clinical data you work with has been
properly de-identified and that you have documented authorization for its
intended use.

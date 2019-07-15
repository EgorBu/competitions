# Plan
## Local validation
* fix local validation
## Generate more data
* Data augmentation (it's not clear how heavy augmentations will affect the result, so start with stable and understandable augmentations)
    * Scale - select part of the image (60~99% - what should be the range?)
    * Rotate - any angle (-180~180 degrees)
    * Mirror - horizontal, vertical, diagonal
    * change average level of image - questionable
* Join images based on depth
## Test time augmentation
* Prepare list of resources
## End-to-end tunable pipeline (hyperopt)
    
# Catan
Scripts for recegnition and analyzis of The Settlers of Catan - multiplayer board game 

### How it works
Script clones image few times, processes and converts each one in particular way (e.g. use gaussian and sobel filter, do tresholding, increase taints, inverse colours), then detects borders of items (like red-counters or forest-fields), marks it and finally merges images to show it.

### What we use
* Python
* openCV
* scikit-image

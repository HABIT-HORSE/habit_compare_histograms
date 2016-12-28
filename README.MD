Author:  Steve North
Author URI:  http://www.cs.nott.ac.uk/~pszsn/
License: AGPLv3 or later
License URI: http://www.gnu.org/licenses/agpl-3.0.en.html
Can: Commercial Use, Modify, Distribute, Place Warranty
Can't: Sublicence, Hold Liable
Must: Include Copyright, Include License, State Changes, Disclose Source

Copyright (c) 2016, The University of Nottingham

Partially based on original (copyright free) code examples by: Adrian Rosebrock: http://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/

Compare histogram of each image found in input_images directory with the histograms of example positive images in comparison_images directory.

Command line argument allows configuring for OpenCV histogram methods. The options are: all, Correlation, Chi-Squared, Intersection or Hellinger

When all methods are used, an 'election' is held to pool the results and to make an overall judgement on whether the object shown in example positive images.
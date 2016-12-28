'''
Author:  Steve North
Author URI:  http://www.cs.nott.ac.uk/~pszsn/
License: AGPLv3 or later
License URI: http://www.gnu.org/licenses/agpl-3.0.en.html
Can: Commercial Use, Modify, Distribute, Place Warranty
Can't: Sublicence, Hold Liable
Must: Include Copyright, Include License, State Changes, Disclose Source

Copyright (c) 2016, The University of Nottingham

Partially based on original (copyright free) code examples by: Adrian Rosebrock: http://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
'''

# import the necessary packages
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2
import os

# Steve paramaters...

# used by loadAllPositiveImagesFoundInDirectory()
comparisonImagesDir = "comparison_images"

# used by loadAllQueryImagesFoundInInputDirectory()
queryImagesDir = "input_images"

# used by loadAllQueryImagesFoundInInputDirectory() and loadAllPositiveImagesFoundInDirectory()
# change this for the required input query input image type... comparison images are currently hardcoded as JPG but don't need to be.
# ... seems OK to compare histograms between different image types.
# OpenCV compatible image formats: bmp, pbm, pgm, ppm, sr, ras, jpeg, jpg, jpe, jp2, tiff, tif, png
imageFileExtension = "jpg"

# used by loadAllQueryImagesFoundInInputDirectory() and parseCommandLineArguments()
openCV_method_mode = "all"

# used by loadAllPositiveImagesFoundInDirectory() and compareQueryImageWithPositiveImagesUsingHistograms()
# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
index = {}
images = {}

###############################  START parseCommandLineArguments #########################################

def parseCommandLineArguments():
	"Parse command line arguments"
	
	global openCV_method_mode   # declare a to be a global

	# construct the argument parser and parse the arguments
	
	ap = argparse.ArgumentParser()
	ap.add_argument("-method", required = False,
		help = "OpenCV comparison method: all, Correlation, Chi-Squared, Intersection or Hellinger")
	args = vars(ap.parse_args())

	#print ("arg = " + args["method"] )

	if args["method"] == "all":
		openCV_method_mode = "all"

	if args["method"] == "Correlation":
		openCV_method_mode = "Correlation"

	if args["method"] == "Chi-Squared":
		openCV_method_mode = "Chi-Squared"

	if args["method"] == "Intersection":
		openCV_method_mode = "Intersection"

	if args["method"] == "Hellinger":
		openCV_method_mode = "Hellinger"
		
	print ("Mode: " + openCV_method_mode)
	
	return

###############################  END parseCommandLineArguments #########################################


###############################  START loadAllPositiveImagesFoundInDirectory #########################################

def loadAllPositiveImagesFoundInDirectory():
	"load All Positive Images Found In Directory"
	# do something
	# loop over the image paths
	
	'''
	# attempt to make it look for any compatible image file]
	
	types = ('*.bmp', '*.pbm', '*.pgm', '*.ppm', '*.sr', '*.ras', '*.jpeg', '*.jpg', '*.jpe', '*.jp2', '*.tiff', '*.tif', '*.png') # the tuple of file types...matching OpenCV compatible image formats
	files_grabbed = []

	for files in types:
		files_grabbed.extend(glob.glob(files))
		
	for imagePath in files_grabbed:
	
	'''
	
	#for imagePath in glob.glob(comparisonImagesDir + "\*." + imageFileExtension):
	for imagePath in glob.glob(comparisonImagesDir + "\*.jpg"):
		# extract the image filename (assumed to be unique) and
		# load the image, updating the images dictionary
		filename = imagePath[imagePath.rfind("\\") + 1:]
		image = cv2.imread(imagePath)
		images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# extract a 3D RGB color histogram from the image,
		# using 8 bins per channel, normalize, and update
		# the index
		hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
		hist = cv2.normalize(hist).flatten()
		index[filename] = hist

	return

###############################  START loadAllQueryImagesFoundInInputDirectory #########################################	
	
	
def loadAllQueryImagesFoundInInputDirectory():
   "load All Query Images FoundIn Input Directory"
   
   global openCV_method_mode   # declare a to be a global
   
   for queryImagePath in glob.glob(queryImagesDir + "\*." + imageFileExtension):	

	print (queryImagePath)
	queryImageFile = cv2.imread(queryImagePath)
	
	# if using this with different code...this is the bit that you would call
	# compareQueryImageWithPositiveImagesUsingHistograms() takes one openCV image object as a query and a string describing an 
	# OpenCV histogram comparison method, calculate comparison values and return true if the elected concensous of all methods 
	# is that the qeury images matches.
	# Acceptable string values for the value openCV_method_mode are: all, Correlation, Chi-Squared, Intersection or Hellinger
	isTargetObject = compareQueryImageWithPositiveImagesUsingHistograms( queryImageFile, openCV_method_mode)
	
	if isTargetObject == True:
		# do something... for example indicate on webcam feed that target objected is detected
		print ("\tDetected...")
	else:
		# do something... for example indicate on webcam feed that target objected is NOT detected
		print ("\tNOT detected...")

	print ("\n") # this is the line return between images!!
   
   return
   
###############################  END loadAllQueryImagesFoundInInputDirectory #########################################
 

 
###############################  START compareQueryImageWithPositiveImagesUsingHistograms #########################################
	
def compareQueryImageWithPositiveImagesUsingHistograms(queryImageFile, openCV_mode):
	"Takes one openCV image object as a querty and a string describing an OpenCV histogram comparison method, calculate comparison values and return true if the elected concensous of all methods is that the qeury images matches"
	
	# Note: acceptable string values for the value openCV_mode are: all, Correlation, Chi-Squared, Intersection or Hellinger
	
	showQueryImage = False;
	showComparisonImagesAndValues = False;

	targetObjectName = "horse"
	
	countPositiveDetectionsForThisImage = 0

	CorrelationThresholdValueForTargetObject = 0.65 # > is object - lower value to increase sensitivity
	ChiSquaredThresholdValueForTargetObject = 7.0 # < is object - raise value to increase sensitivity
	IntersectionThresholdValueForTargetObject = 1.4 # > is object - lower value to increase sensitivity
	HellingerThresholdValueForTargetObject = 0.4 # < is object - raise value to increase sensitivity

	votingScoreThreshold = 50.0
	
	'''
	Correlation - higher value = more accurate match - values up to 1.0? 100%?
	Chi-square - lower value = more accurate match - 0 is a perfect match.
	Intersection - higher value = more accurate match - 24.00 seems to be close to perfect match
	Hellinger (Bhattacharyya) - lower value = more accurate match - 0 is a perfect match.

	CV_COMP_CORREL: [-1;1] where 1 is perfect match and -1 is the worst.
	CV_COMP_CHISQR: [0;+infinty] where 0 is perfect match and mismatch is unbounded (see doc for equation of comparison)
	CV_COMP_INTERSECT: [0;1] (if histograms are normalized) where 1 is perfect match and 0 mismatch.
	CV_COMP_BHATTACHARYYA and CV_COMP_HELLINGER: [0;1] where 0 is perfect match and 1 mismatch.

	Not all methods above actually use proper distance functions. But their results can be made to distance functions:

	for CV_COMP_INTERSECT: dist = 1 - result
	for CV_COMP_CORREL: dist = 2 - (result + 1)

	'''
	
	# METHOD #1: UTILIZING OPENCV
	# initialize OpenCV methods for histogram comparison

	OPENCV_METHODS = []

	if openCV_mode == "all":
		OPENCV_METHODS.append( ("Correlation", cv2.cv.CV_COMP_CORREL) )
		OPENCV_METHODS.append( ("Chi-Squared", cv2.cv.CV_COMP_CHISQR)  )
		OPENCV_METHODS.append( ("Intersection", cv2.cv.CV_COMP_INTERSECT) )
		OPENCV_METHODS.append( ("Hellinger", cv2.cv.CV_COMP_BHATTACHARYYA)  )
		

	if openCV_mode == "Correlation":
		OPENCV_METHODS.append( ("Correlation", cv2.cv.CV_COMP_CORREL))

	if openCV_mode == "Chi-Squared":
		OPENCV_METHODS.append( ("Chi-Squared", cv2.cv.CV_COMP_CHISQR))	

	if openCV_mode == "Intersection":
		OPENCV_METHODS.append( ("Intersection", cv2.cv.CV_COMP_INTERSECT))

	if openCV_mode == "Hellinger":
		OPENCV_METHODS.append( ("Hellinger", cv2.cv.CV_COMP_BHATTACHARYYA))
	
	
	#images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# extract a 3D RGB color histogram from the image,# using 8 bins per channel, normalize, and update
	queryImageHist = cv2.calcHist([queryImageFile], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
	queryImageHist = cv2.normalize(queryImageHist).flatten()
    
		
	# loop over the comparison methods
	for (methodName, method) in OPENCV_METHODS:
		#print ("Method is: " + methodName);
		# initialize the results dictionary and the sort
		# direction
		results = {}
		reverse = False
		#print (methodName)
		# if we are using the correlation or intersection
		# method, then sort the results in reverse order
		if methodName in ("Correlation", "Intersection"):
			reverse = True
		
		# loop over the index
		for (k, hist) in index.items():
			# compute the distance between the two histograms
			# using the method and update the results dictionary
			#d = cv2.compareHist(index[queryImage], hist, method)
			d = cv2.compareHist(queryImageHist, hist, method)
			results[k] = d
			
		# sort the results
		results = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)

		# show the query image
		if showQueryImage:
			fig = plt.figure("Query")
			ax = fig.add_subplot(1, 1, 1)
			ax.imshow(queryImageFile)
			plt.axis("off")
		

		# initialize the results figure
		if showComparisonImagesAndValues:
			fig = plt.figure("Results: %s" % (methodName))
			fig.suptitle(methodName, fontsize = 20)
		
		totalMatchValueForCurrentMethod = 0 #Steve
		# loop over the results
		for (i, (v, k)) in enumerate(results):
			totalMatchValueForCurrentMethod = totalMatchValueForCurrentMethod + v
			# show the result
			if showComparisonImagesAndValues:
				ax = fig.add_subplot(1, len(images), i + 1)
				#ax.set_title("%s: %.2f" % (k, v))
				ax.set_title("%.2f" % (v)) # Steve: edited above so only showing values, rather than long image names, that were overlapping
				#plt.imshow(images[k])
				plt.axis("off")

		averageMatchValueForCurrentMethod = totalMatchValueForCurrentMethod / (i + 1)
		#print ("Sum of values: " + str(totalMatchValueForCurrentMethod) )
		#print ("Number of pix: " + str(i + 1) )
		message = "\t" + methodName + " av. is: " + str(averageMatchValueForCurrentMethod)
		isTargetObject = False
	
		if methodName == "Correlation":
			if averageMatchValueForCurrentMethod > CorrelationThresholdValueForTargetObject: #high averageMatchValueForCurrentMethod is good
				isTargetObject = True
				countPositiveDetectionsForThisImage = countPositiveDetectionsForThisImage + 1
		
		if methodName == "Chi-Squared":
			if averageMatchValueForCurrentMethod < ChiSquaredThresholdValueForTargetObject: #low averageMatchValueForCurrentMethod is good
				isTargetObject = True
				countPositiveDetectionsForThisImage = countPositiveDetectionsForThisImage + 1
		
		if methodName == "Intersection":
			if averageMatchValueForCurrentMethod > IntersectionThresholdValueForTargetObject: #high averageMatchValueForCurrentMethod is good
				isTargetObject = True
				countPositiveDetectionsForThisImage = countPositiveDetectionsForThisImage + 1
		
		if methodName == "Hellinger":
			if averageMatchValueForCurrentMethod < HellingerThresholdValueForTargetObject: #low averageMatchValueForCurrentMethod is good
				isTargetObject = True
				countPositiveDetectionsForThisImage = countPositiveDetectionsForThisImage + 1
		
		if isTargetObject == True:
			message += ": Is a " + targetObjectName + "!"
		else:
			message += ": Is NOT a " + targetObjectName + "!"
		
		print (message)

	# show the OpenCV methods
	plt.show()

	votingScore = float( ( countPositiveDetectionsForThisImage / float(len(OPENCV_METHODS)) * 100 ))
	if votingScore > votingScoreThreshold:
		print ("\tAll election method vote: " + str(votingScore) + "%:" + " Likely a " + targetObjectName);
		isTargetObjectAccordingToCrossMethodVote = True
	else:
		print ("\tAll election method vote: " + str(votingScore) + "%:" + " Likely NOT a " + targetObjectName);
		isTargetObjectAccordingToCrossMethodVote = False
	
	return isTargetObjectAccordingToCrossMethodVote

###############################  END compareQueryImageWithPositiveImagesUsingHistograms #########################################



	
###############################  STUFF STARTS HERE #########################################

parseCommandLineArguments()
loadAllPositiveImagesFoundInDirectory()
loadAllQueryImagesFoundInInputDirectory()

###############################  STUFF ENDS HERE #########################################
	
'''

# STEVE: could try to add the scipy methods and also roll your own method?

# METHOD #2: UTILIZING SCIPY
# initialize the scipy methods to compaute distances
SCIPY_METHODS = (
	("Euclidean", dist.euclidean),
	("Manhattan", dist.cityblock),
	("Chebysev", dist.chebyshev))

# loop over the comparison methods
for (methodName, method) in SCIPY_METHODS:
	# initialize the dictionary dictionary
	results = {}

	# loop over the index
	for (k, hist) in index.items():
		# compute the distance between the two histograms
		# using the method and update the results dictionary
		d = method(index[queryImage], hist)
		results[k] = d

	# sort the results
	results = sorted([(v, k) for (k, v) in results.items()])

	# show the query image
	fig = plt.figure("Query")
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(images[queryImage])
	plt.axis("off")

	# initialize the results figure
	fig = plt.figure("Results: %s" % (methodName))
	fig.suptitle(methodName, fontsize = 20)

	# loop over the results
	for (i, (v, k)) in enumerate(results):
		# show the result
		ax = fig.add_subplot(1, len(images), i + 1)
		ax.set_title("%s: %.2f" % (k, v))
		plt.imshow(images[k])
		plt.axis("off")

# show the SciPy methods
plt.show()

# METHOD #3: ROLL YOUR OWN
def chi2_distance(histA, histB, eps = 1e-10):
	# compute the chi-squared distance
	d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
		for (a, b) in zip(histA, histB)])

	# return the chi-squared distance
	return d

# initialize the results dictionary
results = {}

# loop over the index
for (k, hist) in index.items():
	# compute the distance between the two histograms
	# using the custom chi-squared method, then update
	# the results dictionary
	d = chi2_distance(index[queryImage], hist)
	results[k] = d

# sort the results
results = sorted([(v, k) for (k, v) in results.items()])



# show the query image
fig = plt.figure("Query")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(images[queryImage])
plt.axis("off")

# initialize the results figure
fig = plt.figure("Results: Custom Chi-Squared")
fig.suptitle("Custom Chi-Squared", fontsize = 20)

# loop over the results
for (i, (v, k)) in enumerate(results):
	# show the result
	ax = fig.add_subplot(1, len(images), i + 1)
	ax.set_title("%s: %.2f" % (k, v))
	plt.imshow(images[k])
	plt.axis("off")

# show the custom method
plt.show()

'''
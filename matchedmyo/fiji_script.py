import os
from ij import IJ, WindowManager
from ij.gui import GenericDialog, WaitForUserDialog
from ij.plugin import Macro_Runner
from ij.macro import Functions
from ij.io import FileSaver
from ij.plugin.filter import Rotator
from ij.process import ImageStatistics as IS
from ij.measure import ResultsTable

import subprocess

dev_options = True

### IMPORTANT CODE TO CLEAR COMPILED MODULES
###  FOR DEVELOPMENT PURPOSES
# Use this to recompile Jython modules to class files.
#from sys import modules
#modules.clear()
# Imports of Jython modules are placed below:
#import myModule

# JavaCPP API that contains almost all functions of OpenCV
# import org.bytedeco.javacpp.opencv_core as cv # not working for some reason

### Create context manager for changing directory to matchedmyo directory
class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def auto_resize_angle_measure(img, two_sarcomere_size):
	"""Resizes and measures angle of offset for img based on T-tubules."""
	# Run Fast Fourier Transform on Image to get Power Spectral Density
	IJ.run(img, "FFT", "")

	# Select PSD image.
	fft_title = "FFT of "+img.getTitle()
	fft_img = WindowManager.getImage(fft_title)

	# Threshold the image and create binary mask.
	IJ.setThreshold(151, 255)
	#IJ.setThreshold(135, 255)
	IJ.run(fft_img, "Convert to Mask", "")

	# Set measurements for fitting an ellipse and fit them to data.
	IJ.run(fft_img, "Set Measurements...", "area centroid fit redirect=None decimal=2")
	if dev_options:
		args = "size=10-Infinity show=Ellipses display"
	else:
		args = "size=10-Inifinity"
	IJ.run(fft_img, "Analyze Particles...", args)
	results = ResultsTable.getResultsTable()
	angle_column = results.getColumnIndex("Angle")
	area_column = results.getColumnIndex("Area")
	centroid_x_column = results.getColumnIndex("X")
	centroid_y_column = results.getColumnIndex("Y")
	# Sort by area since we need the largest two ellipses.
	results.sort("Area") 
	# Grab info for the largest two ellipses.
	areas = results.getColumnAsDoubles(area_column)
	angles = results.getColumnAsDoubles(angle_column)
	c_xs = results.getColumnAsDoubles(centroid_x_column)
	c_ys = results.getColumnAsDoubles(centroid_y_column)

	# Calculate offset angle from main ellipse.
	rotation_angle = angles[-1] - 90.
		
	# Measure distance between centroids of two largest ellipses using distance measurement tool to get spatial frequency of T-tubules.
	# Note: This needs to be in pixels since the filter size is in pixels.
	c_x_0 = c_xs[-1]; c_y_0 = c_ys[-1]
	c_x_1 = c_xs[-2]; c_y_1 = c_ys[-2]
	dist_gap = ((c_x_0 - c_x_1)**2 + (c_y_0 - c_y_1)**2)**0.5
	old_two_sarcomere_size = 2 * fft_img.getDimensions()[0] / dist_gap

	# Resize based on measured two sarcomere size
	resize_ratio = float(two_sarcomere_size) / float(old_two_sarcomere_size)
	old_width = img.width
	old_height = img.height
	new_width = resize_ratio * old_width
	new_height = resize_ratio * old_height
	resize_args = "width={} height={} depth=1 constrain average interpolation=Bicubic".format(
		new_width,
		new_height
	)
	IJ.run(img, "Size...", resize_args)

	return rotation_angle

##############################################################################################################################
### Begin Actual Code
##############################################################################################################################

def run():
	### Default arguments 
	two_sarcomere_size = 25 # determined by filter used.
	rotation_angle = 0.0
	
	### Get the image we'd like to work with.
	# Don't need to ask where to save the intermediate image since it'll just be saved in the matchedmyo folder anyway
	#   may change this though. In case they want to keep that image around.
	this_img = WindowManager.getCurrentImage()
	if this_img == None:
		ud = WaitForUserDialog(
			"Please select the image you would like to analyze."
		)
		ud.show()
		this_img = WindowManager.getCurrentImage()
	img_name = this_img.getTitle()
	
	matchedmyo_path = "/home/AD/dfco222/scratchMarx/matchedmyo/" # this would be grabbed from the prompt
	gd = GenericDialog("Preprocessing Options")
	gd.addCheckbox("Automatic Resizing/Rotation", True)
	gd.addCheckbox("CLAHE", True)
	gd.addCheckbox("Normalization to Transverse Tubules", True)
	gd.showDialog()
	if gd.wasCanceled():
		return
	auto_resize_rotate = gd.getNextBoolean()
	clahe = gd.getNextBoolean()
	normalize = gd.getNextBoolean()

	if auto_resize_rotate:
		# Clear selection so it takes FFT of full image

		rotation_angle = auto_resize_angle_measure(this_img, two_sarcomere_size)

	if clahe:
		clahe_args = "blocksize={} histogram=256 maximum=3 mask=*None* fast_(less_accurate)".format(two_sarcomere_size)
		IJ.run(this_img, "Enhance Local Contrast (CLAHE)", clahe_args) 

	if normalize:
		# Ask the user to select a subsection of the image that looks healthy-ish.
		ud = WaitForUserDialog(
			"Please select a subsection exhibiting a measure of healthy TT structure using the Rectangle tool.\n"
			+" Only a single cell or several are needed.\n\n"
			+" Press 'OK' when finished."
		)
		ud.show()
		IJ.setTool("rectangle")
		
		# Duplicate the selected subsection.
		selection = this_img.crop()
		IJ.run(selection, "Duplicate...", "title=subsection.tif")

		# Grab the subsection image and rotate it.
		selection = WindowManager.getImage("subsection.tif")
		IJ.run(selection, "Rotate...", "angle={} grid=1 interpolation=Bicubic enlarge".format(rotation_angle))

		# Ask the user to select a bounding box that contains only tubules
		# NOTE: Need to get rid of initial selection since it's the entire image and it's annoying to click out of
		IJ.setTool("rectangle")
		IJ.run(selection, "Select None", "")
		ud = WaitForUserDialog(
			"Select a subsection of the image that contains only tubules and no membrane."
		)
		ud.show()

		# Grab the subsection ImagePlus
		selection = WindowManager.getCurrentImage()
		this_window = WindowManager.getActiveWindow()
		selection_small = selection.crop()
		IJ.run(selection, "Close", "")
		
		# NOTE: May not actually display this depending on how the macros work
		IJ.run(selection_small, "Duplicate...", "title=subsection_small.tif")
		
		# Smooth the selection using the single TT filter.
		# NOTE: It won't read in so we're just going to hard code it in since it's simple
		tt_filt_row = "0 0 0 1 1 1 1 1 1 0 0 0 0\n"
		tt_filt = ""
		for i in range(21):
			tt_filt += tt_filt_row
		IJ.run("Convolve...", "text1=["+tt_filt+"] normalize")
		
		# Segment out the TTs from the 'gaps' using Gaussian Adaptive Thresholding.
		selection_small = WindowManager.getImage("subsection_small.tif")
		IJ.run(selection_small, "Duplicate...", "title=thresholded.tif")
		threshed = WindowManager.getImage("thresholded.tif")
		IJ.run(threshed, "Auto Local Threshold", "method=Bernsen radius=7 parameter_1=1 parameter_2=0 white")
		
		# Select the TTs from the thresholded image.
		IJ.run(threshed, "Create Selection", "")
		tt_selection = WindowManager.getImage("subsection_small.tif")
		IJ.selectWindow("thresholded.tif")
		IJ.selectWindow("subsection_small.tif")
		IJ.run(tt_selection, "Restore Selection", "")
		
		# Get TT intensity statistics.
		stat_options = IS.MEAN | IS.MIN_MAX | IS.STD_DEV
		stats = IS.getStatistics(tt_selection.getProcessor(), stat_options, selection_small.getCalibration())
		# Calculate pixel ceiling intensity value based on heuristic.
		# TODO: Add a catch for data type overflow.
		pixel_ceiling = stats.mean + 3 * stats.stdDev
		print "px ceil:", pixel_ceiling

		# Invert selection to get inter-sarcomeric gap intensity statistics.
		IJ.run(tt_selection, "Make Inverse", "")
		stat_options = IS.MEAN | IS.MIN_MAX | IS.STD_DEV
		stats = IS.getStatistics(tt_selection.getProcessor(), stat_options, selection_small.getCalibration())
		# Calculate pixel floor intensity value based on heuristic.
		pixel_floor = stats.mean - stats.stdDev
		# TODO: Add a catch for data type underflow.
		print "px floor:", pixel_floor

		# Threshold original image based on these values.
		IJ.selectWindow(this_img.getTitle())
		IJ.run(this_img, "Select All", "")
		IJ.setMinAndMax(pixel_floor, pixel_ceiling)
		IJ.run(this_img, "Apply LUT", "")

	## Ask if it is acceptable.
	gd = GenericDialog("Acceptable?")
	gd.addMessage(
		"If the preprocessed image is acceptable for analysis, hit 'OK' to being analysis.\n"
		+" If the image is unacceptable or an error occurred, hit 'Cancel'"
	)
	gd.showDialog()
	if gd.wasCanceled():
		return
	
	## Save the preprocessed image.
	imp = IJ.getImage()
	fs = FileSaver(imp)
	img_save_dir = matchedmyo_path+"myoimages/" # actually get from user at some point
	img_file_path = img_save_dir + img_name[:-4] + "_preprocessed.tif"
	if os.path.exists(img_save_dir) and os.path.isdir(img_save_dir):
		print "Saving image as:", img_file_path
		if os.path.exists(img_file_path):
			# use dialog box to ask if they want to overwrite
			gd = GenericDialog("Overwrite?")
			gd.addMessage("A file exists with the specified path, \"{}\". Would you like to overwrite it?".format(img_file_path))
			gd.enableYesNoCancel()
			gd.showDialog()
			if gd.wasCanceled():
				return
		elif fs.saveAsTiff(img_file_path):
			print "Preprocessed image saved successfully at:", '"'+img_file_path+'"'
	else:
		print "Folder does not exist or is not a folder!"

	### Create the YAML file containing the parameters for classification
	## Ask user for YAML input
	gd = GenericDialog("YAML Input")
	gd.addStringField("imageName", img_file_path, 50)
	#gd.addStringField("maskName", "None")
	gd.addStringField("outputParams_fileRoot", img_file_path[:-4], 50)
	gd.addStringField("outputParams_fileType", "tif")
	gd.addNumericField("outputParams_dpi", 300, 0)
	gd.addCheckbox("outputParams_saveHitsArray", False)
	gd.addStringField("outputParams_csvFile", matchedmyo_path+"results/")
	gd.addCheckbox("TT Filtering", True); gd.addToSameRow()
	gd.addCheckbox("LT Filtering", True); gd.addToSameRow()
	gd.addCheckbox("TA Filtering", True)
	gd.addNumericField("scopeResolutions_x", 5.0, 3); gd.addToSameRow()
	gd.addNumericField("scopeResolutions_y", 5.0, 3)
	gd.addMessage("Enter in filter rotation angles separated by commas.")
	gd.addStringField("", "-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25", 50) 
	gd.addCheckbox("returnAngles", False)
	gd.addCheckbox("returnPastedFilter", True)
	gd.showDialog()
	if gd.wasCanceled():
		return

	strings = [st.text for st in gd.getStringFields()]
	#if strings[1] == "None" or "":
	#	strings[1] = None
	nums = [float(num.text) for num in gd.getNumericFields()]
	nums[0] = int(nums[0]) # Have to make sure the dpi variable is an integer
	checks = [str(bool(boo.state)) for boo in gd.getCheckboxes()]
	iter_argument = ','.join([str(float(it) - rotation_angle) for it in strings[4].split(',')])
	string_block = """imageName: {0[0]}
outputParams:
  fileRoot: {0[1]}
  fileType: {0[2]}
  dpi: {1[0]}
  saveHitsArray: {2[0]}
  csvFile: {0[3]}
preprocess: False
filterTypes:
  TT: {2[1]}
  LT: {2[2]}
  TA: {2[3]}
scopeResolutions:
  x: {1[1]}
  y: {1[2]}
iters: [{3}]
returnAngles: {2[4]}
returnPastedFilter: {2[5]}""".format(strings, nums, checks, iter_argument)
	im_title = this_img.getTitle()
	with cd(matchedmyo_path):
		yaml_file_path = "./YAML_files/"+im_title[:-4]+".yml"
		with open("./YAML_files/"+im_title[:-4]+".yml", "w") as ym:
			ym.write(string_block)
		print "Wrote YAML file to:", matchedmyo_path + yaml_file_path[2:]

	### Run the matchedmyo code on the preprocessed image
	with cd(matchedmyo_path):
		#os.chdir(matchedmyo_path)
		#subprocess.call(["python3", matchedmyo_path+"matchedmyo.py", "fullValidation"])
		subprocess.call(["python3", "matchedmyo.py", "run", "--yamlFile", yaml_file_path])

run()
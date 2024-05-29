The entire point of my study is that it explores the effects of different 
sizes of data sets. 

- i have left all my files in there, you can just delete the 2 datasets and the saved models.

Changes to make to the System for functionality:

tensor, keras, os, matplotlib, numpy are required

For testing purposes:
	- Uncomment Line 150 (there is also a commented guide)

For Training: 
if you want to use small dataset: 

	- change line 108 parameter with 'data_dir_small'
	- change String in line 90 to "SmallSavedModel"
	- change String in line 129 to "SmallSavedModel"

if you want to use Large dataset:

	- change line 108 parameter with 'data_dir_large'
	- change String in line 90 to "LargeSavedModel"
	- change String in line 129 to "LargeSavedModel"

alternatively, if you have your own folder of images that will be used for training:
	
	- add directory to String in line 105
	- change line 108 parameter with 'data_set'


if the images you want to test with are png: 
	- on line 11, add png to that list of extentions 

if you want to test without training:
	- comment out line108
	- uncomment line 150

# Photo-Gradients
This program is intended to allow the sorting and searching of images. One of
the main features is that it allows you to make image gradients arrangements.
It also allows you to search through images by colors within the image, lens
used during image creation, camera body used etc.

# Requirements
Python 2 with installs/support for the following
subprocess, operator, math, random, time, argparse, Counter, multiprocessing,
threading, itertools, base64

numpy
cv2
matplotlib
colorspacious
ortools.constraint_solver (routing_enums_pb2 and pywrapcp)
sklearn.cluster
tqdm
cPickle

# Instructions
Within the Photo-Gradients folder, make an empty Images folder and an empty Saves folder.
Place image .jpg files in the Images folder, and run with `python Main.py`.

If a save file is found, type `y` to use it or `n` to ignore and load all
images within the Images folder. If a save file was generated with images that
are no longer within the Images folder, you must use `n`. Additionally, to load
more images, use `n`. Otherwise, using `y` will save you time. After entering `y`,
the loading image filenames will appear, allowing you to see what files will be used.
If an image is not found it will be indicated next to the file name, and the program
will exit.

Replace api-key on line 35 of Search.py with your personal applicable api key.

After loading of images, you will be prompted to save the loaded ordering data.
Type `y` to save or `n` otherwise.

Next, you will see some filenames, as well as extracted image metadata load up.

You will be prompted on whether to search or view a gradient. Type `g` to see
the gradient, `s` to use Microsoft Azure Search, and then enter a color as a
search query (things like pink, red, dark red etc will work best). You will then
see the number of found results. Enter the number to display, no greater than the
number found.

#Credits

Image-ExifTool-11.28
Using Image-ExifTool-11 allows lens based, date captured or camera body based
search terms.
https://www.sno.phy.queensu.ca/~phil/exiftool/#links

Microsoft Azure Search for searching images
colorspacious for color space conversions and distances.
ortools for gradient formations.
sklearn.cluster for color quantization on the images
tqdm for load bars
cPickle for saving

#License
Copyright 2019 Mitchell Pavlak

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

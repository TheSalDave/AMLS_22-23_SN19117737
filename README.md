# ELEC0134-Applied Machine Learning Systems I Final Assignment Student number 19117737

## Overview

This repository contains the code and documentation for the final assignment of the course ELEC0134-Applied Machine Learning Systems I. The assignment consists of four machine learning tasks: two binary classification tasks and two multiclass classification tasks. The datasets used for these tasks are the `celeba` dataset and the `cartoon_set` dataset.

The folders A1,A2,B1,B2 contain files for the corresponding tasks. Please note that B1 and B2 were accidentally swapped due to confusion when reading in the columns from the label files. For main.py this was corrected by reading in B1 model for B2 and vice versa. Each python notebook has clear names indicating which model is trained in it. LabMethods include the traditional Machine Learning models from the labs whilst other files are training CNNs using pytorch.

To run the final models, run main.py. As RegnetY16GF model used in A1 was too large main.py needs to retrain the model for 6 epochs which is likely to take a lot of time. For other tasks the trained model was uploaded and is used for those tasks. The output of main.py presents the confusion matrices of all tasks as one figure.

## Task A: Binary Classification on `celeba` dataset

### Task A1: Gender Detection

* Description: Classify images as male or female

            Train   Valid   Test
SVM		    CV	    92.4
RF		    CV	    87
CNN		    95.53	93.333
AlexNet		94.85	91.354
ConvTiny	94.71	93.438
RegnetY16GF	96.55	94.53	95.94

Final model: RegnetY16GF

### Task A2: Emotion Detection

* Description: Classify images as smiling or not smiling

            Train   Valid   Test
RFC		    CV	    88.9* 	                    *but only after discarding 205 images
SVM		    CV	    89.16*                      *but only after discarding 205 images
CNN		    90.58	88.5	88.646
AlexNet		83.33	78.125
ConvTiny	83.01	80.625
RegnetY16GF	87.75	85.4

Final model: Manual CNN

## Task B: Multiclass Classification on `cartoon_set` dataset

### Task B1: Face Shape Recognition

* Description: Classify face shapes into 5 categories

            Train   Valid   Test
CNN		    99.83	99.65
CNNNoRot	100	    100	    100

Final model: Manual CNN without rotation of images

### Task B2: Eye Color Recognition

* Description: Classify eye colors into 5 categories

            Train   Valid   Test
CNN		    87.25	84.6	83.84
AlexNet		78.03	75.9
ConvTiny	79.06	78.9
RegnetY16GF	NA	    NA

Final model: Manual CNN without rotation of images

# Mask_OPC_UNet
This is the project of AI4IC, a model to optimize the litho output mask

# Introduction and Background
## Lithography Process
Lithography is a fundamental process in semiconductor manufacturing where geometric patterns from a photomask are transferred onto a photosensitive resist on a wafer. The process follows the
sequence: Layout → Mask → Wafer. The primary goal is to ensure that the patterns on the layout are accurately transferred to the wafer through an optical system, modeled by the Hopkins diffraction model, with minimal distortion. A significant challenge in this process is the mismatch
between the lithography system and the feature sizes of the devices.
## Optical Proximity Correction (OPC)
Optical Proximity Correction (OPC) is a technique used to modify the mask patterns to achieve the
optimal lithographic process window. Due to imperfections in the optical system and diffraction
effects, the patterns on the resist may not perfectly match those on the mask. OPC involves
correcting the mask patterns to ensure that the patterns on the resist meet the desired specifications.
Model-based OPC uses optical and resist models to predict the patterns on the resist after exposure.
## Inverse Lithography Technology (ILT)
Traditional lithography involves generating a wafer image from a given optimized mask pattern
using a lithography simulation process. Inverse Lithography Technology (ILT) aims to find the
optimal mask pattern that produces a wafer image as close as possible to the desired target pattern.
This involves solving the inverse problem. ILT can provide a more precise lithographic process
window but is computationally intensive and time-consuming.
# OPC Algorithm
## ILT
ILT is an end-to-end correction process where the layout image is input, and the mask image is
output. This iterative process, similar to an autoencoder, uses gradient descent to optimize the mask.
Given a dataset, with the target layout as input and the optimized mask as the label, a simple UNet
can be used for supervised learning. However, direct use of UNet resulted in unacceptable errors,necessitating a correction process to enhance performance. This led to the introduction of an
unsupervised neural-network training approach.
## Neural-ILT
Neural-ILT comprises three submodules:
1. Pre-trained UNet Module: Converts the layout to the mask.
2. ILT Correction Layer: Minimizes the inverse lithography loss using gradient descent.
3. Mask Complexity Refinement Layer: Removes redundant features to refine mask
complexity.
The core of the ILT correction layer is to minimize the differences between the layout and the mask
through gradient descent. The Mask Complexity Refinement Layer addresses issues such as edge
glitches, redundant contours, and isolated curvilinear stains that may arise during ILT, potentially
impacting the maximum process window.
# Algorithm Implementation
## ILT Algorithm
1. Input: Target layout.
2. Initial Mask Generation: Use the pre-trained UNet to generate an initial mask.
3. ILT Correction: Apply the ILT correction layer to iteratively minimize the difference
between the layout and the mask using gradient descent.
4. Mask Refinement: Use the Mask Complexity Refinement Layer to remove redundant
features and refine the mask.
5. Output: Optimized mask.
## Neural-ILT Algorithm
1. Pre-trained UNet Module:
• Train a UNet model on the dataset with target layout as input and the optimized mask
as the label.
• Use the trained UNet to generate an initial mask from the layout.
2. ILT Correction Layer:
• Implement a gradient descent algorithm to iteratively minimize the inverse
lithography loss.• Update the mask in each iteration to reduce the difference between the layout and the
mask.
3. Mask Complexity Refinement Layer:
• Identify and remove edge glitches, redundant contours, and isolated curvilinear
stains.
• Refine the mask to ensure it meets the desired specifications and does not impact the
process window.
# Code execution
## Running the OPC Model
1. Navigate to the project directory:/proj1/Lithosim/lithosim.
2. Execute the OPC model script to generate the target mask: 
$ python OPC_model.py
## Evaluating the Mask
1. Navigate to the project root directory:/proj1.
2. Run the evaluation script to assess the mask and generate the COST score:
$ ./auto_eval.sh

# Reference
Neural-ILT 2.0: Migrating ILT to Domain-Specific and Multitask-Enabled Neural Network

# Bildgebung Task 2: Registration

## Mono-modal Image Registration with B-Spline algorithm

Data available at: https://cloud.imi.uni-luebeck.de/s/o7LyCbJCie8fQ3B?openfile=true

Framework used: https://simpleitk.org

https://www.sciencedirect.com/topics/computer-science/spline-registration

# Parameters
mesh-size, 
intensity-metric (mattes),
optimizer and its params

# To-Dos
- landmarks (look up point-based registration slides, caution: rigid vs. non-rigid)
- keypoints (plot on fixed image)
- understand sitk.Compose
- define registration method which loops over all input images
- save fixed, moving and overlay for all input images
- understand task: why training and test sets?

# Questions
- register per slice or per image? 208 per nifti object or just 1?
- if and how to use landmarks?
- how to use keypoints?
- mesh size: 4-6?
- optimizer: LBFGS (Limited memory Broyden Fletcher Goldfarb Shannon minimization) or Gradient Descent?
- interpolator: linear or b-spline?
- similarity metric: mattes?


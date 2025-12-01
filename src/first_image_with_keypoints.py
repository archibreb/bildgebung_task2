import SimpleITK as sitk
import numpy as np
import csv
import os
import matplotlib.pyplot as plt

# ===============================
# Paths
# ===============================
BASE = "/Users/rebeccaarchibald/bildgebung_task2/data/LungCT"
OUT  = "/Users/rebeccaarchibald/bildgebung_task2/test"
os.makedirs(OUT, exist_ok=True)

# ===============================
# Utility: Load landmarks from CSV
# ===============================
def load_keypoints_csv(path):
    pts = np.loadtxt(path, delimiter=',')
    return pts.flatten().tolist()

# ===============================
# Registration callback
# ===============================
def command_iteration(method):
    print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f}")

# ===============================
# Load images and masks
# ===============================
fixed  = sitk.ReadImage(f"{BASE}/imagesTr/LungCT_0001_0000.nii.gz", sitk.sitkFloat32)
moving = sitk.ReadImage(f"{BASE}/imagesTr/LungCT_0001_0001.nii.gz", sitk.sitkFloat32)

fixed_mask  = sitk.ReadImage(f"{BASE}/masksTr/LungCT_0001_0000.nii.gz", sitk.sitkUInt8)
moving_mask = sitk.ReadImage(f"{BASE}/masksTr/LungCT_0001_0001.nii.gz", sitk.sitkUInt8)

# Ensure masks are binary
fixed_mask  = fixed_mask > 0
moving_mask = moving_mask > 0

# ===============================
# Load landmarks
# ===============================
fixed_points_list  = load_keypoints_csv(f"{BASE}/keypointsTr/LungCT_0001_0000.csv")
moving_points_list = load_keypoints_csv(f"{BASE}/keypointsTr/LungCT_0001_0001.csv")

# ===============================
# Landmark-based affine initialization
# ===============================
affine_tx = sitk.LandmarkBasedTransformInitializer(
    sitk.AffineTransform(fixed.GetDimension()),
    fixed_points_list,
    moving_points_list
)
print("Affine transform initialized from landmarks.")

# ===============================
# Initialize B-Spline transform
# ===============================
mesh_size = [4] * fixed.GetDimension()  # Adjust mesh size for lungs
bspline_tx = sitk.BSplineTransformInitializer(fixed, mesh_size)
print("B-Spline transform initialized.")

# ===============================
# Registration setup
# ===============================
R = sitk.ImageRegistrationMethod()

# Intensity metric
R.SetMetricAsMattesMutualInformation(50)

# Masks
R.SetMetricFixedMask(fixed_mask)
R.SetMetricMovingMask(moving_mask)

# Interpolator
R.SetInterpolator(sitk.sitkLinear)

# Optimizer
R.SetOptimizerAsLBFGSB(
    gradientConvergenceTolerance=1e-5,
    numberOfIterations=10,
    maximumNumberOfCorrections=5,
    maximumNumberOfFunctionEvaluations=2000,
    costFunctionConvergenceFactor=1e7,
)

# Set transforms: B-spline for refinement, affine for initialization
R.SetInitialTransform(bspline_tx, inPlace=False)
R.SetMovingInitialTransform(affine_tx)

R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

# ===============================
# Run registration
# ===============================
print("\nRunning B-spline registration ...")
outTx = R.Execute(fixed, moving)

print("\n------- REGISTRATION DONE -------")
print("Optimizer stop:", R.GetOptimizerStopConditionDescription())
print("Iterations:", R.GetOptimizerIteration())
print("Final metric value:", R.GetMetricValue())

# ===============================
# Save transform
# ===============================
sitk.WriteTransform(outTx, os.path.join(OUT, "bspline_affine_keypoints_masks.tfm"))

# ===============================
# Warp moving image
# ===============================
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetTransform(outTx)
resampler.SetDefaultPixelValue(0)
warped_moving = resampler.Execute(moving)
sitk.WriteImage(warped_moving, os.path.join(OUT, "registered_with_keypoints.nii.gz"))

# ===============================
# Overlay image for visualization
# ===============================
fixed_u8  = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
moving_u8 = sitk.Cast(sitk.RescaleIntensity(warped_moving), sitk.sitkUInt8)
overlay = sitk.Compose(fixed_u8, moving_u8, (fixed_u8 + moving_u8) // 2)
sitk.WriteImage(overlay, os.path.join(OUT, "overlay_rgb_with_keypoints.nii.gz"))

# ===============================
# Quick matplotlib view (overlay_with_landmarks)
# ===============================
slice_z = overlay.GetSize()[2] // 2
overlay_np = sitk.GetArrayFromImage(overlay)
plt.imshow(overlay_np[slice_z], cmap='gray')
plt.title("Overlay slice with keypoints")
plt.axis('off')
plt.show()

# ===============================
# Quick matplotlib view (registered)
# ===============================
slice_z2 = warped_moving.GetSize()[2] // 2
registered_np = sitk.GetArrayFromImage(warped_moving)
plt.imshow(registered_np[slice_z2], cmap='gray')
plt.title("Registered slice with keypoints")
plt.axis('off')
plt.show()

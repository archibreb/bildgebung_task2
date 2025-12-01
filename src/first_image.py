import SimpleITK as sitk
import os
import matplotlib.pyplot as plt

test = "/Users/rebeccaarchibald/bildgebung_task2/test"

def command_iteration(method):
    """Callback invoked when the optimization has an iteration"""
    print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f}")


# ---------------------------------------------------
# Load images
# ---------------------------------------------------
fixed = sitk.ReadImage(
    "/Users/rebeccaarchibald/bildgebung_task2/data/LungCT/imagesTr/LungCT_0001_0000.nii.gz",
    sitk.sitkFloat32
)
moving = sitk.ReadImage(
    "/Users/rebeccaarchibald/bildgebung_task2/data/LungCT/imagesTr/LungCT_0001_0001.nii.gz",
    sitk.sitkFloat32
)

# ---------------------------------------------------
# Load masks (binary)
# ---------------------------------------------------
fixed_mask = sitk.ReadImage(
    "/Users/rebeccaarchibald/bildgebung_task2/data/LungCT/masksTr/LungCT_0001_0000.nii.gz",
    sitk.sitkUInt8
)
moving_mask = sitk.ReadImage(
    "/Users/rebeccaarchibald/bildgebung_task2/data/LungCT/masksTr/LungCT_0001_0001.nii.gz",
    sitk.sitkUInt8
)

# Ensure masks match image geometry (VERY important)
fixed_mask = sitk.Resample(fixed_mask, fixed, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
moving_mask = sitk.Resample(moving_mask, moving, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

# Binarize masks if needed
fixed_mask = fixed_mask > 0
moving_mask = moving_mask > 0


# ---------------------------------------------------
# Initialize B-spline transform
# ---------------------------------------------------
transformDomainMeshSize = [4] * moving.GetDimension()
tx = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize)

print("Initial Parameters:")
print(tx.GetParameters())


# ---------------------------------------------------
# Registration setup
# ---------------------------------------------------
R = sitk.ImageRegistrationMethod()

R.SetMetricAsMattesMutualInformation(50)

# ← MASKS ARE SET HERE
R.SetMetricFixedMask(fixed_mask)
R.SetMetricMovingMask(moving_mask)

R.SetOptimizerAsLBFGSB(
    gradientConvergenceTolerance=1e-5,
    numberOfIterations=10, # was 100, decrease to 10 for testing
    maximumNumberOfCorrections=5,
    maximumNumberOfFunctionEvaluations=1000,
    costFunctionConvergenceFactor=1e7
)

R.SetInitialTransform(tx, inPlace=False)
R.SetInterpolator(sitk.sitkLinear)

R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))


# ---------------------------------------------------
# Run registration
# ---------------------------------------------------
outTx = R.Execute(fixed, moving)

print("-------")
print(outTx)
print("Optimizer stop condition:", R.GetOptimizerStopConditionDescription())
print("Iteration:", R.GetOptimizerIteration())
print("Metric value:", R.GetMetricValue())


# ---------------------------------------------------
# Save transform
# ---------------------------------------------------
sitk.WriteTransform(outTx, os.path.join(test, "bspline.tfm"))


# ---------------------------------------------------
# Apply transform to moving image
# ---------------------------------------------------
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(100)
resampler.SetTransform(outTx)

out = resampler.Execute(moving)

# Create composition image for visualization
simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
cimg = sitk.Compose(simg1, simg2, simg1 // 2 + simg2 // 2)

print({"fixed": fixed, "moving": moving, "composition": cimg})

slice_z = cimg.GetSize()[2] // 2
plt.imshow(sitk.GetArrayFromImage(cimg)[slice_z], cmap='gray')
plt.title("Overlay")
plt.axis('off')
plt.show()

slice_z2 = fixed.GetSize()[2] // 2
plt.imshow(sitk.GetArrayFromImage(fixed)[slice_z2], cmap='gray')
plt.title("Fixed")
plt.axis('off')
plt.show()

slice_z3 = moving.GetSize()[2] // 2
plt.imshow(sitk.GetArrayFromImage(moving)[slice_z3], cmap='gray')
plt.title("Moving")
plt.axis('off')
plt.show()
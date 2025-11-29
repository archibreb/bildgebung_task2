import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

print(sitk.Version())
print("Import of simpleITK framework successful")

training_images_dir = '/Users/rebeccaarchibald/bildgebung_task2/data/LungCT/imagesTr'
training_masks_dir  = '/Users/rebeccaarchibald/bildgebung_task2/data/LungCT/masksTr'

test_images_dir = '/Users/rebeccaarchibald/bildgebung_task2/data/LungCT/imagesTs'
test_masks_dir  = '/Users/rebeccaarchibald/bildgebung_task2/data/LungCT/masksTs'

training_image_files = sorted([f for f in os.listdir(training_images_dir) if f.endswith(('.nii', '.nii.gz'))])
training_mask_files  = sorted([f for f in os.listdir(training_masks_dir)  if f.endswith(('.nii', '.nii.gz'))])

test_image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith(('.nii', '.nii.gz'))])
test_mask_files  = sorted([f for f in os.listdir(test_masks_dir)  if f.endswith(('.nii', '.nii.gz'))])

def load_training_images_and_masks(training_image_files, training_mask_files):
    # Lists to store SimpleITK images
    training_images = []
    training_masks  = []

    for img_file, mask_file in zip(training_image_files, training_mask_files):
        # Full paths
        img_path = os.path.join(training_images_dir, img_file)
        mask_path = os.path.join(training_masks_dir, mask_file)
        
        # Read images
        img = sitk.ReadImage(img_path, sitk.sitkFloat32)
        mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)  # masks as integer labels
        
        # Append to lists
        training_images.append(img)
        training_masks.append(mask)

    print(f"Training: Loaded {len(training_images)} images and {len(training_masks)} masks")
    return training_images, training_masks

def load_test_images_and_masks(test_image_files, test_mask_files):
    # Lists to store SimpleITK images
    test_images = []
    test_masks  = []

    for img_file, mask_file in zip(test_image_files, test_mask_files):
        # Full paths
        img_path = os.path.join(test_images_dir, img_file)
        mask_path = os.path.join(test_masks_dir, mask_file)
        
        # Read images
        img = sitk.ReadImage(img_path, sitk.sitkFloat32)
        mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)  # masks as integer labels
        
        # Append to lists
        test_images.append(img)
        test_masks.append(mask)

    print(f"Test: Loaded {len(test_images)} images and {len(test_masks)} masks")
    return test_images, test_masks

training_images, training_masks = load_training_images_and_masks(training_image_files=training_image_files,
                               training_mask_files=training_mask_files)

test_images, test_masks = load_test_images_and_masks(test_image_files=test_image_files,
                           test_mask_files=test_mask_files)
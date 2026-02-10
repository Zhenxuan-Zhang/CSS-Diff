import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
random.seed(100)
from torchvision import io
from torchvision import transforms  # Import torchvision transforms


def reindex(file_path):
    """Pad the slice index of the filename with zeros so that it can be sorted
    lexicographically while keeping the numerical order.
    """
    directory, filename = os.path.split(file_path)
    name, ext = os.path.splitext(filename)

    x_value = int(name.split('_')[-1])
    x_padded = f"{x_value:10d}"

    new_filename = name.replace(f"_{x_value}", f"_{x_padded}")
    new_file_path = os.path.join(directory, new_filename + ext)
    return new_file_path


class AlignedMaskDataset(BaseDataset):
    """Dataset class for *aligned* A/B pairs with corresponding masks.

    Directory structure (same as the unaligned version):
    - <dataroot>/<phase>_A
    - <dataroot>/<phase>_B
    - <dataroot>/<phase>_maskA
    - <dataroot>/<phase>_maskB

    For training we assume that slice *n* in domain A corresponds to slice *n*
    in domain B (after sorting). If the two domains contain a different number
    of slices, the shorter one is cycled so every A always has a B.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        # ------------------------------------------------------------------
        # Locate image folders
        # ------------------------------------------------------------------
        self.dir_A = os.path.join(opt.dataroot, f"{opt.phase}_{opt.Aclass}")
        self.dir_B = os.path.join(opt.dataroot, f"{opt.phase}_{opt.Bclass}")
        self.dir_maskA = os.path.join(opt.dataroot, f"{opt.phase}_mask{opt.Aclass}")
        self.dir_maskB = os.path.join(opt.dataroot, f"{opt.phase}_mask{opt.Bclass}")

        # ------------------------------------------------------------------
        # Make dataset lists (sorted so that indices line up)
        # ------------------------------------------------------------------
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.maskA_paths = sorted(make_dataset(self.dir_maskA, opt.max_dataset_size))
        self.maskB_paths = sorted(make_dataset(self.dir_maskB, opt.max_dataset_size))

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        # Optionally keep only half of each modality (first half of A, last half of B)
        if opt.half:
            self.A_size //= 2
            self.B_size //= 2
            self.A_paths = self.A_paths[: self.A_size]
            self.maskA_paths = self.maskA_paths[: self.A_size]
            self.B_paths = self.B_paths[-self.B_size :]
            self.maskB_paths = self.maskB_paths[-self.B_size :]

        # ------------------------------------------------------------------
        # Build transforms
        # ------------------------------------------------------------------
        btoA = opt.direction == "BtoA"
        input_nc = opt.output_nc if btoA else opt.input_nc
        output_nc = opt.input_nc if btoA else opt.output_nc
        self.transform_A = get_transform(opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(opt, grayscale=(output_nc == 1))
        self.resize_transform = transforms.Resize((224, 224))  # Not used but kept for backward compatibility

    # ------------------------------------------------------------------
    # Standard PyTorch Dataset interface
    # ------------------------------------------------------------------
    def __getitem__(self, index):
        """Return one aligned A/B pair and the associated masks."""
        # Domain A ----------------------------------------------------------
        index_A = index % self.A_size
        A_path = self.A_paths[index_A]

        # Domain B (aligned with A) ----------------------------------------
        # If domain B is shorter, wrap around with modulo so every A gets a B.
        index_B = index % self.B_size
        B_path = self.B_paths[index_B]

        # Corresponding masks ----------------------------------------------
        maskA_path = self.maskA_paths[index_A]
        maskB_path = self.maskB_paths[index_B]

        # Load images -------------------------------------------------------
        A_img = io.read_image(A_path)
        B_img = io.read_image(B_path)
        A_mask = io.read_image(maskA_path)
        B_mask = io.read_image(maskB_path)

        # Apply joint transformations -------------------------------------
        A, A_mask = self.transform_A(A_img, A_mask)
        B, B_mask = self.transform_B(B_img, B_mask)

        return {
            "A": A,
            "B": B,
            "A_paths": A_path,
            "B_paths": B_path,
            "A_mask": A_mask,
            "B_mask": B_mask,
        }

    def __len__(self):
        """Return the maximum length of the two domains so __getitem__ never
        raises an IndexError when one domain is shorter than the other.
        """
        return max(self.A_size, self.B_size)

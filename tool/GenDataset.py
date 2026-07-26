# -*- coding: utf-8 -*-
import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tool import custom_transforms as tr

class Stage1_InferDataset(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        self.object = self.path_label()

    def __getitem__(self, index):
        fn = self.object[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return fn.split('/')[-1][:-4], img
        
    def __len__(self):
        return len(self.object)
        
    def path_label(self):
        path_list = []
        for root, dirname, filename in os.walk(self.data_path):
            for f in filename:
                image_path = os.path.join(root, f)
                path_list.append(image_path)
        return path_list


def _build_validation_dataloader(config):
    """
    Build the Stage-1 validation dataloader.

    Responsibilities
    ----------------
    1. Create validation transforms.
    2. Build the inference dataset.
    3. Build the validation dataloader.

    Parameters
    ----------
    config : Stage1Config
        Stage-1 configuration.

    Returns
    -------
    torch.utils.data.DataLoader
        Validation dataloader.
    """

    # ------------------------------------------------------------
    # Validation Transform
    # ------------------------------------------------------------

    validation_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # ------------------------------------------------------------
    # Validation Dataset
    # ------------------------------------------------------------

    validation_dataset = Stage1_InferDataset(
        data_path=config.testroot,
        transform=validation_transform,
    )

    # ------------------------------------------------------------
    # Validation DataLoader
    # ------------------------------------------------------------

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return validation_loader

class Stage1_TrainDataset(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None, dataset=None):
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset
        self.object = self.path_label()
        

    def __getitem__(self, index):
        fn, label = self.object[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return fn.split('/')[-1][:-4], img, label
        
    def __len__(self):
        return len(self.object)
        
    def path_label(self):
        path_label = []
        for root, dirname, filename in os.walk(self.data_path):
            for f in filename:
                image_path = os.path.join(root, f)
                fname = f[:-4]
                ##  Extract the image-level label from the filename
                ##  LUAD-HistoSeg   : 'Image-name-of-BCSS'+'+index'+'[abcd]'.png
                ##  BCSS-WSSS       : 'patient_ID'+'_x-axis'+'_y-axis'+'[a b c d]'.png
                label_str = fname.split(']')[0].split('[')[-1]
                if self.dataset == 'luad':
                    image_label = torch.Tensor([int(label_str[0]),int(label_str[2]),int(label_str[4]),int(label_str[6])])
                elif self.dataset == 'bcss':
                    image_label = torch.Tensor([int(label_str[0]),int(label_str[1]),int(label_str[2]),int(label_str[3])])
                path_label.append((image_path, image_label))
        return path_label


class Stage1CurriculumDataset(Dataset):
    """
    Stage-1 dataset used for curriculum iterations (iteration > 0).

    Returns
    -------
    filename
    image
    ground_truth_label
    pseudo_label
    """

    def __init__(
        self,
        data_path,
        pseudo_label_file,
        transform=None,
        target_transform=None,
        dataset=None,
        use_ground_truth_as_pseudo=False,
    ):

        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset
        self.use_ground_truth_as_pseudo = use_ground_truth_as_pseudo

        # --------------------------------------------------------
        # Load pseudo labels
        # --------------------------------------------------------
        if not self.use_ground_truth_as_pseudo:
            with open(pseudo_label_file, "r") as f:
                self.pseudo_labels = json.load(f)

        # --------------------------------------------------------
        # Build dataset
        # --------------------------------------------------------

        self.samples = self.path_label()

    def __getitem__(self, index):

        image_path, gt_label = self.samples[index]

        filename = os.path.basename(image_path)
        image_id = os.path.splitext(filename)[0]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            gt_label = self.target_transform(gt_label)

        # --------------------------------------------------------
        # Retrieve pseudo label
        # --------------------------------------------------------
        if self.use_ground_truth_as_pseudo:

            pseudo_label = gt_label.clone()

        else:

            pseudo_label = torch.tensor(
                self.pseudo_labels[image_id]["pseudo_label"],
                dtype=torch.float32,
            )


        return (
            image_id,
            image,
            gt_label,
            pseudo_label,
        )

    def __len__(self):
        return len(self.samples)

    def path_label(self):

        samples = []

        for root, _, files in os.walk(self.data_path):

            for file in files:

                image_path = os.path.join(root, file)

                filename = os.path.splitext(file)[0]

                label_str = filename.split("]")[0].split("[")[-1]

                if self.dataset == "luad":

                    gt_label = torch.tensor([
                        int(label_str[0]),
                        int(label_str[2]),
                        int(label_str[4]),
                        int(label_str[6]),
                    ], dtype=torch.float32)

                elif self.dataset == "bcss":

                    gt_label = torch.tensor([
                        int(label_str[0]),
                        int(label_str[1]),
                        int(label_str[2]),
                        int(label_str[3]),
                    ], dtype=torch.float32)

                else:
                    raise ValueError(
                        f"Unsupported dataset: {self.dataset}"
                    )

                samples.append((image_path, gt_label))

        return samples



class Stage2_Dataset(Dataset):
    def __init__(self, args, base_dir, split):

        super().__init__()
        self._base_dir = base_dir
        self.split = split

        if self.split   == "train":
            self._image_dir     = os.path.join(self._base_dir, 'train5/')
            self._cat_dir       = os.path.join(self._base_dir, 'train_PM/PM_bn7/')
            self._cat_dir_a     = os.path.join(self._base_dir, 'train_PM/PM_b5_2')
            self._cat_dir_b     = os.path.join(self._base_dir, 'train_PM/PM_b4_5')
        elif self.split == 'val':
            self._image_dir = os.path.join(self._base_dir, 'val/img/')
            self._cat_dir   = os.path.join(self._base_dir, 'val/mask/')
        elif self.split == 'test':
            self._image_dir = os.path.join(self._base_dir, 'test/img/')
            self._cat_dir   = os.path.join(self._base_dir, 'test/mask/')
        self.args = args
        self.filenames = [os.path.splitext(file)[0] for file in os.listdir(self._image_dir) if not file.startswith('.')]
        self.images = [os.path.join(self._image_dir, fn + '.png') for fn in self.filenames]
        self.categories = [os.path.join(self._cat_dir, fn + '.png') for fn in self.filenames]
        if self.split == "train":
            self.categories_a = [os.path.join(self._cat_dir_a, fn + '.png') for fn in self.filenames]
            self.categories_b = [os.path.join(self._cat_dir_b, fn + '.png') for fn in self.filenames]
        assert (len(self.images) == len(self.categories))
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.split == "train": 
            _img, _target, _target_a, _target_b = self._make_img_gt_point_pair(index)
            sample = {'image': _img, 'label': _target, 'label_a': _target_a, 'label_b': _target_b}
        elif (self.split == 'val') or (self.split == 'test'):
            _img, _target, = self._make_img_gt_point_pair(index)
            sample = {'image': _img, 'label': _target}
            image_dir = self.images[index]
        if self.split == "train":
            return self.transform_tr_ab(sample)
        elif (self.split == 'val') or (self.split == 'test'):
            return self.transform_val(sample), image_dir
    def _make_img_gt_point_pair(self, index):
        if self.split == "train": 
            _img = Image.open(self.images[index]).convert('RGB')
            _target = Image.open(self.categories[index])
            _target_a = Image.open(self.categories_a[index])
            _target_b = Image.open(self.categories_b[index])
            return _img,_target,_target_a,_target_b
        elif (self.split == 'val') or (self.split == 'test'):
            _img = Image.open(self.images[index]).convert('RGB')
            _target = Image.open(self.categories[index])
            return _img,_target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomGaussianBlur(),
            tr.Normalize(),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_tr_ab(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip_ab(),
            tr.RandomGaussianBlur_ab(),
            tr.Normalize_ab(),
            tr.ToTensor_ab()])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.Normalize(),
            tr.ToTensor()])
        return composed_transforms(sample)

    def __str__(self):
        return None

def make_data_loader(args, **kwargs):

    train_set   = Stage2_Dataset(args, base_dir=args.dataroot, split='train')
    val_set     = Stage2_Dataset(args, base_dir=args.dataroot, split='val')
    test_set    = Stage2_Dataset(args, base_dir=args.dataroot, split='test')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader

def make_data_loader_v2(args, **kwargs):

    train_set   = Stage2_Dataset(args, base_dir=args.dataroot, split='train')
    val_set     = Stage2_Dataset(args, base_dir=args.dataroot, split='val')
    test_set    = Stage2_Dataset(args, base_dir=args.dataroot, split='test')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader


class Stage2DatasetV2(Dataset):
    """
    Stage-2 Semantic Segmentation Dataset.

    The dataset paths are supplied through the configuration instead
    of being hardcoded inside the dataset.
    """

    def __init__(
        self,
        config,
        split,
    ):

        super().__init__()

        self.config = config
        self.split = split.lower()

        # ------------------------------------------------------------
        # Resolve directories from configuration
        # ------------------------------------------------------------

        if self.split == "train":

            self.image_dir = config.stage2_train_image_dir
            self.label_dir = config.stage2_train_label_dir
            self.label_dir_a = config.stage2_train_label_dir_a
            self.label_dir_b = config.stage2_train_label_dir_b

        elif self.split == "val":

            self.image_dir = config.stage2_val_image_dir
            self.label_dir = config.stage2_val_label_dir

        elif self.split == "test":

            self.image_dir = config.stage2_test_image_dir
            self.label_dir = config.stage2_test_label_dir

        else:

            raise ValueError(f"Unknown split: {split}")

        # ------------------------------------------------------------
        # File lists
        # ------------------------------------------------------------

        self.filenames = sorted(
            os.path.splitext(file)[0]
            for file in os.listdir(self.image_dir)
            if not file.startswith(".")
        )

        self.images = [
            os.path.join(self.image_dir, f"{name}.png")
            for name in self.filenames
        ]

        self.labels = [
            os.path.join(self.label_dir, f"{name}.png")
            for name in self.filenames
        ]

        if self.split == "train":

            self.labels_a = [
                os.path.join(self.label_dir_a, f"{name}.png")
                for name in self.filenames
            ]

            self.labels_b = [
                os.path.join(self.label_dir_b, f"{name}.png")
                for name in self.filenames
            ]

        assert len(self.images) == len(self.labels)

        print(
            f"Number of images in {self.split}: {len(self.images)}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        if self.split == "train":

            image, label, label_a, label_b = self._load_sample(index)

            sample = {
                "image": image,
                "label": label,
                "label_a": label_a,
                "label_b": label_b,
            }

            return self.transform_tr_ab(sample)

        else:

            image, label = self._load_sample(index)

            sample = {
                "image": image,
                "label": label,
            }

            return self.transform_val(sample), self.images[index]

    def _load_sample(self, index):

        image = Image.open(
            self.images[index]
        ).convert("RGB")

        label = Image.open(
            self.labels[index]
        )

        if self.split == "train":

            label_a = Image.open(
                self.labels_a[index]
            )

            label_b = Image.open(
                self.labels_b[index]
            )

            return image, label, label_a, label_b

        return image, label

    def transform_tr(self, sample):

        composed = transforms.Compose(
            [
                tr.RandomHorizontalFlip(),
                tr.RandomGaussianBlur(),
                tr.Normalize(),
                tr.ToTensor(),
            ]
        )

        return composed(sample)

    def transform_tr_ab(self, sample):

        composed = transforms.Compose(
            [
                tr.RandomHorizontalFlip_ab(),
                tr.RandomGaussianBlur_ab(),
                tr.Normalize_ab(),
                tr.ToTensor_ab(),
            ]
        )

        return composed(sample)

    def transform_val(self, sample):

        composed = transforms.Compose(
            [
                tr.Normalize(),
                tr.ToTensor(),
            ]
        )

        return composed(sample)

    def __str__(self):
        return self.split
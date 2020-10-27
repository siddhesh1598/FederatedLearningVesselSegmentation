#import
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import syft as sy
import cv2
from imutils import paths

# DataGenerator class
class DataGenerator(Dataset):

    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return [image, mask]


class DatasetGenerator:
	"""
		Args:
			data_dirs = {"DRIVE"= [images_dir, masks_dir],
						"STARE" = [images_dir, masks_dir]}
		
		allImages = {"DRIVE" = list of all images in DRIVE dataset,
				"STARE" = list of all images in STARE dataset}

		allMasks = {"DRIVE" = list of all masks in DRIVE dataset,
					"STARE" = list of all masks in STARE dataset}
	"""

	def __init__(self, data_dirs, args):
		
		self.args = args
		self.allImages = {}
		self.allMasks = {}

		for (key, value) in data_dirs.items():
			images_dir, masks_dir = value[0], value[1]

			self.allImages[key] = list(paths.list_images(images_dir))
			self.allMasks[key] = list(paths.list_images(masks_dir))


	def generateTrainDataset(self, key):
		"""
			Generate Training Data for given key i.e. "DRIVE"/"STARE"
			Args:
				key: Name of the Dataset 
			Return:
				DataLoader for the required dataset
		"""
		images = list()
		masks = list()

		imagePaths = self.allImages[key]
		maskPaths = self.allMasks[key]

		imagePaths.sort()
		maskPaths.sort()

		# resize the images and masks
		for (imagePath, maskPath) in zip(imagePaths, maskPaths):
			image = cv2.resize(cv2.imread(imagePath), self.args.image_size)
			mask = cv2.resize(cv2.imread(maskPath), self.args.image_size)
			images.append(image)
			masks.append(mask)

		images = images[:self.args.split]
		masks = masks[:self.args.split]

		# generate the dataset loader for the image and mask dataset
		print("[INFO] Generating Training Dataset Loader for {}...".format(key))

		print("[INFO] Total number of images for {} training dataset: {}".format(
								key, len(images)))
		print("[INFO] Total number of masks for {} training dataset: {}".format(
								key, len(masks)))

		transform = transforms.Compose([
			transforms.ToTensor(),
		])

		train_set = DataGenerator(images, masks, transform = transform)
		trainDataLoader = DataLoader(train_set, batch_size=self.args.batch_size, 
			shuffle=True)

		return trainDataLoader


	def generateFederatedDataset(self):
		"""
			Generate Federated Data
			Args:
				workers (tuple): 
					the workers to distribute the dataset 
			Return:
				DataLoader for the required dataset
		"""
		images = list()
		masks = list()

		for (key, value) in self.allImages.items():
			value = value[:self.args.split]
			value.sort()
			for imagePath in value:
				image = cv2.resize(cv2.imread(imagePath), self.args.image_size)
				images.append(image)

		for (key, value) in self.allMasks.items():
			value = value[:self.args.split]
			value.sort()
			for maskPath in value:
				mask = cv2.resize(cv2.imread(maskPath), self.args.image_size)
				masks.append(mask)

		# generate the federated  dataset loader for the image and mask dataset
		print("[INFO] Generating Federated Dataset Loader...")

		print("[INFO] Total number of images for FEDERATED training dataset: {}".format(
								len(images)))
		print("[INFO] Total number of masks for FEDERATED training dataset: {}".format(
								len(masks)))
		
		transform = transforms.Compose([
			transforms.ToTensor(),
		])

		hook = sy.TorchHook(torch)
		bob = sy.VirtualWorker(hook, id="bob")
		alice = sy.VirtualWorker(hook, id="alice")
		jon = sy.VirtualWorker(hook, id="jon") # secore worker

		workers = (bob, alice)

		federated_set = DataGenerator(images, masks, transform = transform)
		federatedDataLoader = sy.FederatedDataLoader(
				federated_set.federate(workers), 
				batch_size = self.args.fed_batch_size, 
				shuffle = True)

		return (federatedDataLoader, (bob, alice, jon))


	def generateTestDataset(self):
		"""
			Generate Test Data
			Args:
				None 
			Return:
				DataLoader for the required dataset
		"""
		images = list()
		masks = list()

		for (key, value) in self.allImages.items():
			value = value[self.args.split:]
			value.sort()
			for imagePath in value:
				image = cv2.resize(cv2.imread(imagePath), self.args.image_size)
				images.append(image)

		for (key, value) in self.allMasks.items():
			value = value[self.args.split:]
			value.sort()
			for maskPath in value:
				mask = cv2.resize(cv2.imread(maskPath), self.args.image_size)
				masks.append(mask)

		# generate the dataset loader for the image and mask dataset
		print("[INFO] Generating Test Dataset Loader...")

		print("[INFO] Total number of images for test dataset: {}".format(len(images)))
		print("[INFO] Total number of masks for test dataset: {}".format(len(masks)))

		transform = transforms.Compose([
			transforms.ToTensor(),
		])

		test_set = DataGenerator(images, masks, transform = transform)
		testDataLoader = DataLoader(test_set, batch_size=self.args.test_batch_size, 
			shuffle=True)

		return testDataLoader


	




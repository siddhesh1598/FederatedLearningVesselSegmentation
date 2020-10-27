# import
import torch
import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt

# arguments
class Arguments():

	def __init__(self):
		self.image_size = (224, 224)
		self.batch_size = 16
		self.test_batch_size = 8
		self.fed_batch_size = 32
		self.split = 16
		self.epochs = 100
		self.lr = 0.0001
		self.momentum = 0.5
		self.log_interval = 10
		self.save_model = False

# Dice Loss
def dice_loss(pred, target, smooth = 1.):
	pred = pred.contiguous()
	target = target.contiguous()    

	intersection = (pred * target).sum(dim=2).sum(dim=2)

	loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

	return loss.mean()


def calculate_loss(pred, target):
	pred = torch.sigmoid(pred)
	loss = dice_loss(pred, target)

	return loss


def calculate_iou(pred, target):
	intersection = np.logical_and(target, pred)
	union = np.logical_or(target, pred)
	iou_score = np.sum(intersection) / np.sum(union)

	return iou_score


# aggregate
def aggregate(model, model_alice, model_bob, secure_worker):

    model_alice.move(secure_worker)
    model_bob.move(secure_worker)

    for (_, paramA), (_, paramB), (_, param) in zip(model_alice.named_parameters(), 
                                            model_bob.named_parameters(),
                                            model.named_parameters()):
        param.data = (((paramA.data + paramB.data) / 2).get())

    return model


# Plot images
def plot_images(dataLoader, index = 0):
	inputs, masks = next(iter(dataLoader))
	print("Image Shape: ", inputs.shape, "\nMask Shape: ", masks.shape)

	input = inputs[index].numpy().transpose((1, 2, 0))
	mask = masks[index].numpy().transpose((1, 2, 0))	

	f, axarr = plt.subplots(1,2)
	axarr[0].imshow(input)
	axarr[1].imshow(mask)


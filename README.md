# FederatedLearningVesselSegmentation

![alt text](https://github.com/siddhesh1598/FederatedLearningVesselSegmentation/blob/main/thumbnail.png?raw=true)

An implementation of Federated Learning on the [DRIVE](https://drive.grand-challenge.org/) and [STARE](https://cecas.clemson.edu/~ahoover/stare/) datasets. The datasets contain *images of retinal blood vessels* and *masks segmenting the blood vessels*. The code implements **UNet Model** individually on the DRIVE and the STARE dataset and then combining both the datasets and perform Federated Learning over them, providing a layer of *privacy*.

## Technical Concepts

**Federated Learning:** Federated learning *(also known as collaborative learning)* is a machine learning technique that trains an algorithm across multiple decentralized edge devices or servers holding local data samples, without exchanging them. <br>
More information can be found [here](https://blog.openmined.org/tag/federated-learning/)

## Getting Started

Clone the project repository to your local machine, then follow up with the steps as required. <br>
Download the datasets from here: <br>
[DRIVE](https://drive.google.com/file/d/1MP-L3ecSzLaW8LKKcDrAoF60xOCBjlip/view?usp=sharing) <br>
[STARE](https://drive.google.com/file/d/1L1h-7UTUMPzGHqnZMPaS20QeghPP8gDK/view?usp=sharing) <br>

## Requirements

After cloning the repository, install the necessary requirements for the project.
```
pip install -r requirements.txt
```

## Implementation

To perform Vessel Segmentation, **UNet** model is used which is present in the *model.py* file. The *generator.py* contains the data loader class to generate the datasets for individual DRIVE and STARE datasets as well as the combined Federated dataset. **Federated Learning** is performed using **OpenMined's syft library**. Two workers are used to distribute the dataset and a third worker acts as a secure worker for aggrigation of the models. The models are trained for 100 epochs. The trained models are then evaluated using the **IoU (Intersection over Union)** metrics, where the higher value implies more accuracy of the model.

### Training model on DRIVE dataset
![DRIVE loss](https://github.com/siddhesh1598/FederatedLearningVesselSegmentation/blob/main/stats/DRIVE_loss.png?raw=true)

### Training model on STARE dataset
![STARE loss](https://github.com/siddhesh1598/FederatedLearningVesselSegmentation/blob/main/stats/STARE_loss.png?raw=true)

### Training model on FEDERATED dataset
![FEDERATED loss](https://github.com/siddhesh1598/FederatedLearningVesselSegmentation/blob/main/stats/FEDERATED_loss.png?raw=true)

### Test results
![Test results](https://github.com/siddhesh1598/FederatedLearningVesselSegmentation/blob/main/stats/TestScore_IoU.png?raw=true) <br>
The test results show that the model trained on combined DRIVE and STARE dataset gives more accuracy as compared to the models trained on individual DRIVE and STARE datasets. This allows organizations to work on each other's datasets without actually sharing them, providing a layer of privacy. <br>
Also, these results are for dataset with about 40 images. The power of Privacy Preserving algotrithms increase as the dataset increases. 


## Authors

* **Siddhesh Shinde** - *Initial work* - [SiddheshShinde](https://github.com/siddhesh1598)


## Acknowledgments

* DRIVE Dataset: [DRIVE](https://drive.grand-challenge.org/) <br>
* STARE Dataset: [STARE](https://cecas.clemson.edu/~ahoover/stare/) <br>

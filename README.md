# Coordinating Two UR5 Robots for a pick and place task
<p align="center">
<img src="https://github.com/murtazabasu/Identify-Pneumothorax-Disease-using-UNet/tree/master/media/media.PNG" width="650">

Pneumothorax can be caused by a blunt chest injury, damage from underlying lung disease, or most horrifying—it may occur for no obvious reason at all. On some occasions, a collapsed lung can be a life-threatening event. Pneumothorax is usually diagnosed by a radiologist on a chest x-ray, and can sometimes be very difficult to confirm. An accurate AI algorithm to detect pneumothorax would be useful in a lot of clinical scenarios. AI could be used to triage chest radiographs for priority interpretation, or to provide a more confident diagnosis for non-radiologists.

In this repository a model is developed using UNet with the help of Convolution Neural Networks that takes a chest x-ray image as input and predicts whether the given image has a pneumothorax or not. The ``SIIM_ACR` class in `data.py` performs preprocessing and augmentation on the images and the masks from the dataset before feeding it to the model for training. The dataset is available at `https://www.kaggle.com/vbookshelf/pneumothorax-chest-xray-images-and-masks`.

#### Criterion for using this repository:
- This project was tested on Python 3.7 using Conda Distribution
- Choose the `num_workers` for parallelizing the training and validation dataset using pytorch based on your system specification.
- If your cuda memory gets full, try resize the images using `img_resize` argument and also try decreasing the batch size for training.
    

#### References
- https://github.com/abhishekkrthakur/approachingalmost

- https://www.kaggle.com/vbookshelf/pneumothorax-chest-xray-images-and-masks
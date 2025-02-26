# Climbing Footwork Image Classification - Experiment and Compare With Various Models

### Project Summary:

This project builds various Machine Learning models and compare their performance to classify **climbing footwork** images into **3 categories**: **heelhook**, **toehook**, or **others**. 

I utilized the data that I collected by writing a web image scraper script, followed by duplicate removing and format adjusting etc.

The Machine Learning models were created using **Tensorflow** and **Keras*.

---

### Dataset Description:

![](./visuals/climbing_footwork_classification_img.png?raw=true) 

- Collected the data by writing a web image scraper script and searching the keywords in 5 different languages: `English`, `Japanese`, `Spanish`, `French`, `Chinese`.
- Remove duplicated images by writing a script and implementing pHash. I chose this approach as I wanted to remove the same images with different sizes and image qualities.
- Cleaned the data by converting the formats of all the images to be **png* type.
- Data breakdown:

    Label	       |  number
    :-----------:|:--------:
    heel_hook    | 106
    toe_hook     | 76
    others       | 124
- Since there were not many data available and the data was imbalanced. I have also implemented transfer learning and class weight computation to minigate it.
  
---

## Comparison of Machine Learning Models:

I built various Machine Learning models through transfer learning with different base models, here is the list of all the base models I experimented:
1) `MobileNetV3Large`
2) `EfficientNetB2`
3) `ResNet50`
4) `VGG16`

### Configurations:

I kept the following configurations the same to be consistent across all the base model experiments: 

#### Machine Learning Model Architecture:

The models utilizes transfer learning composed of the following elements:
- Each base model with `imagenet` weights
- `Dense` layer with `relu` activation function
- `Dropout` layer with `0.2`
- `Dense` layer with `relu` activation function
- `Dropout` layer with `0.2`
- `Dense` layer with `softmax` activation function

#### Data Augmentation and Resizing and Rescaling:

For the training dataset, I applied the following Data Augmentation to avoid overfitting:

Data Augmentation - 
```
data_augmentation = keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2),
  tf.keras.layers.RandomZoom(0.2),
  tf.keras.layers.RandomHeight(0.2),
  tf.keras.layers.RandomWidth(0.2),
], name="data_augmentation")
```

I also applied the following Resizing and Rescaling:

Resizing and Rescaling - 
```
resize_and_rescale = tf.keras.Sequential([
  tf.keras.layers.Resizing(224,224),
  tf.keras.layers.Rescaling(1./255),
])
```

#### Training Hyperparameters:

* Epochs: `100` with `EarlyStopping` with the `patience` of `30`
  
* Optimizer: `Adam`

* Learning Rate: `0.00001`

* Batch size: `32`
 
### Analysis of the Results:

#### Loss:

From the loss plots drawn below, you can see that training loss is decreasing while validation loss is not. This shows that the models are overfitting. I have experimented with different configrations as well, but it seems that this overfitting is due to the lack of data available. We need much more data to improve the loss, but since this is a niche nature of the topic, the data availability was limited.

MobileNetV3Large           | EfficientNetB2           
:-------------------------:|:-------------------------:
![](./visuals/MobileNetV3Large/mobilenet_loss.png?raw=true)  | ![](./visuals/EfficientNetB2/efficientnet_loss.png?raw=true) 

| ResNet50                  |  VGG16
|:-------------------------:|:-------------------------:
| ![](./visuals/ResNet50/resnet_loss.png?raw=true) | ![](./visuals/VGG16/vgg_loss.png?raw=true)

### Accuracy:

Similar to the loss, accuracy also improved for training but not for validation for all of the models for the same reason mentioend as above.

MobileNetV3Large           | EfficientNetB2          
:-------------------------:|:-------------------------:
![](./visuals/MobileNetV3Large/mobilenet_accuracy.png?raw=true)  | ![](./visuals/EfficientNetB2/efficientnet_accuracy.png?raw=true)  

| ResNet50                  |  VGG16
|:-------------------------:|:-------------------------:
| ![](./visuals/ResNet50/resnet_accuracy.png?raw=true)  | ![](./visuals/VGG16/vgg_accuracy.png?raw=true) 

### Example Inference:

The green color represents the correctly predicted images while the red color represents the mistakenly predicted images. All the models contains a number of the red/ incorrect predictions. This shows that if the model is overfitted, it cannot accurately predict the categories.  

MobileNetV3Large           | EfficientNetB2       
:-------------------------:|:-------------------------:
![](./visuals/MobileNetV3Large/mobilenet_inference.png?raw=true)   | ![](./visuals/EfficientNetB2/efficientnet_inference.png?raw=true) 

| ResNet50                  |  VGG16
|:-------------------------:|:-------------------------:
| ![](./visuals/ResNet50/resnet_inference.png?raw=true) | ![](./visuals/VGG16/vgg_inference.png?raw=true)

### Classification Report:

Which model did the best ? While all four models had not so great performance due to the lack of data, the model based on the **ResNet50** has the highest **Macro Average F1 Score** of **0.42**. I chose the **Macro F1 Score** to compare the performance in order to ensure that all classes are considered.

MobileNetV3Large           | EfficientNetB2           
:-------------------------:|:-------------------------:
![](./visuals/MobileNetV3Large/mobilenet_classification_report.png?raw=true)  | ![](./visuals/EfficientNetB2/efficientnet_classification_report.png?raw=true) 

| ResNet50                  |  VGG16
|:-------------------------:|:-------------------------:
| ![](./visuals/ResNet50/resnet_classification_report.png?raw=true)  | ![](./visuals/VGG16/vgg_classification_report.png?raw=true) 

### Conclusion:

This was a great experience of choosing a niche topic and collecting the data on my own from the web. I cleaned and formatted the data to be easily utilized for my Machine Learning models. I also explored different configrations and base models for trainsfer learning. My biggest learning was that building a model requires a lot more data, and if I want to build a model, I need to ensure to have access to enough data. For the future, I would love to come up with a way to automatically collect a large enough amount of data for my Machine Learning trainings.
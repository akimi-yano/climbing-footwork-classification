# Climbing Footwork Image Classification - Experiment and Compare With Various Models

### Project Summary:

This project builds various Machine Learning models and compare their performance to classify **climbing footwork** images into **3 categories**: **heelhook**, **toehook**, or **others**. 

I utilized the data that I collected by writing a web image scraper script, followed by duplicate removing and format adjusting etc.

The Machine Learning models were created using **Tensorflow** and **Keras*.

---

### Dataset Description:

TODO: ADD THE IMAGE(s) HERE

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
 
### Fingings:

#### Loss:

TODO: ADD THE IMAGE(s) HERE

MobileNetV3Large           | EfficientNetB2           | ResNet50                  |  VGG16
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](./visuals/lung_classification_loss.png?raw=true)  | ![](./visuals/lung_classification_loss.png?raw=true) | ![](./visuals/lung_classification_loss.png?raw=true) | ![](./visuals/lung_classification_loss.png?raw=true)

### Accuracy:

TODO: ADD THE IMAGE(s) HERE

MobileNetV3Large           | EfficientNetB2           | ResNet50                  |  VGG16
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](./visuals/lung_classification_loss.png?raw=true)  | ![](./visuals/lung_classification_loss.png?raw=true) | ![](./visuals/lung_classification_loss.png?raw=true) | ![](./visuals/lung_classification_loss.png?raw=true)

### Example Inference:

TODO: ADD THE IMAGE(s) HERE

MobileNetV3Large           | EfficientNetB2           | ResNet50                  |  VGG16
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](./visuals/lung_classification_loss.png?raw=true)  | ![](./visuals/lung_classification_loss.png?raw=true) | ![](./visuals/lung_classification_loss.png?raw=true) | ![](./visuals/lung_classification_loss.png?raw=true)

### Classification Report:

TODO: ADD THE IMAGE(s) HERE

MobileNetV3Large           | EfficientNetB2           | ResNet50                  |  VGG16
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](./visuals/lung_classification_loss.png?raw=true)  | ![](./visuals/lung_classification_loss.png?raw=true) | ![](./visuals/lung_classification_loss.png?raw=true) | ![](./visuals/lung_classification_loss.png?raw=true)

### Conclusion:


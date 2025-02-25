# Climbing Footwork Image Classification - Experiment and Compare With Various Models

### Project Summary:

This project builds various Machine Learning models and compare their performance to classify **climbing footwork** images into **3 categories**: **heelhook**, **toehook**, or **others**. 

I utilized the data that I collected by writing a web image scraper script, followed by duplicate removing and format adjusting etc.

The Machine Learning models were created using **Tensorflow** and **Keras*.

---

### Dataset Description:

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

### Machine Learning Model Comparison:

The models utilizes transfer learning and I used various models as a base: 1) 





---

### Machine Learning Model Architecture:

The models utilizes transfer learning composed of the following elements:
- `MobileNetV3Small` with `imagenet` weights
- `Dense` layer with `relu` activation function
- `BatchNormalization`
- `Dropout` layer
- `Dense` layer with `softmax` activation function

---

### Data Augmentation:

For the training dataset, I applied the following data augmentation to avoid overfitting:

- `zca_epsilon` = `0.0`
- `horizontal_flip` = `True`

---

### Training Hyperparameters:

* Epochs: `30`
  
* Optimizer: `Adam`

* Learning Rate: `0.001`

* Batch size: `64`

---

### Loss:

![](./visuals/lung_classification_loss.png?raw=true)

---

### Accuracy:

![](./visuals/lung_classification_accuracy.png?raw=true)

---

### Confusion Matrix without Normalization:

![](./visuals/lung_classification_confusion_matrix_without_normalization.png?raw=true)

---

### Normalized Confusion Matrix:

![](./visuals/lung_classification_normalized_confusion_matrix.png?raw=true)

---

### Result:

This implementation was able to achieve the following accuracy scores:

- Training Accuracy: `0.9561` 
- Validation Accuracy: `0.9241`

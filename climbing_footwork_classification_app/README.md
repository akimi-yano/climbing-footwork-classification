# Climbing Footwork Classification App

### Deployed: https://climbing-footwork-classification.streamlit.app/

### Demo Video: https://youtu.be/JxlRRbaMjQo

## Introduction:

This app is an app to classify the climbing footwork technique using Computer Vision technology !

- Built and compared various Machine Learning models for image classification from data collection and processing to model training.
- Realized the end-to-end user experience by deploying the models with containerized backend api and hosting the frontend code.

## How to Use the App:

Please try out the climbing footwork classification by using any climbing photo !

1. Upload an image of a person climbing using the upload button.
2. Select a model using the dropdown button.
3. Once the image is uploaded and model is selected, the 'Classify' button will appear. Click the 'Classify' button and see the displayed classification result with the confidence score.

## How to Run the App Locally:

If you are interested in running this app locally, here are the steps:

### Backend
1. `cd` into the `backend` directry. 
2. Specify the port to run locally in the `app.py` file.
3. Run `python app.py`.

### Frontend 
1. `cd` into the `frontend` directory.
2. Point to the port where the backend code is running in the `app.py` file.
3. Run `streamlit run app.py`.

## Technologies:

### Machine Learning Models
To train the machine learning models, I collected image data by writing the web scrapers and searching in multiple spoken languages such as Japanese, Chinese, Spanish, French, and English. After that, I conducted various image processing by adjusting the format of the images and removing duplicated images through **similarity detection** techniques. To build performant models, I experimented by creating a number of models using transfer learning including **resnet50**, **mobilenet v3 large**, **efficientnet b2** and **vgg16**. I have also explored with **GPT 4 Turbo LLM** with **zero-shot** approach, which was really interesting !

### Frontend
The frontend was built using Python and Streamlit, and it was deployed on Streamlit Cloud Community.

### Backend
The backend was built using Python and Flask, and it was deployed on Google Cloud Platform's Cloud Run after containerized with Docker.

### Thank you !
The experience to build an AI powered application end-to-end was wonderful, and it was a great learning experience. I would love to continue extending on top this idea and build a even more powerful and comprehensive app with much better model's performance. Thank you for your time taking a look at my project !
# Handwritten Digit Recognition

This project implements a Handwritten Digit Recognition system using the MNIST dataset and TensorFlow/Keras. The model is trained to recognize digits (0-9) from grayscale images.

## Project Overview

Handwritten Digit Recognition is a common machine learning problem that involves classifying handwritten digits into one of the 10 classes (0-9). This project demonstrates the process of training a neural network on the MNIST dataset and using the trained model to predict digits from new images.

## Features

- **Data Loading and Normalization**: Loads the MNIST dataset and normalizes the pixel values to a range of 0 to 1.
- **Model Architecture**: Uses a simple neural network with two hidden layers and a softmax output layer.
- **Training**: Trains the model on the MNIST training data.
- **Model Saving and Loading**: Saves the trained model to disk and provides functionality to load the model for future predictions.
- **Prediction**: Predicts handwritten digits from new images using the trained model.

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- OpenCV
- Matplotlib

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   ```

2. Install the required packages:

   ```bash
   pip install tensorflow numpy opencv-python matplotlib
   ```

## Usage

1. **Training the Model**:

   Uncomment the following lines in the `main.py` script to train the model:

   ```python
   mnist = tf.keras.datasets.mnist
   (x_train, y_train), (x_test, y_test) = mnist.load_data()

   x_train = tf.keras.utils.normalize(x_train, axis=1)
   x_test = tf.keras.utils.normalize(x_test, axis=1)

   model = tf.keras.models.Sequential()
   model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
   model.add(tf.keras.layers.Dense(128, activation='relu'))
   model.add(tf.keras.layers.Dense(128, activation='relu'))
   model.add(tf.keras.layers.Dense(10, activation='softmax'))

   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

   model.fit(x_train, y_train, epochs=10)

   model.save('handwritten.keras')
   ```

2. **Loading and Using the Model**:

   The `main.py` script includes code to load the trained model and predict digits from new images stored in the `digits` directory:

   ```python
   model = tf.keras.models.load_model('handwritten.keras')

   image_number = 1
   while os.path.isfile(f"digits/digit{image_number}.png"):
       try:
           img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
           img = np.invert(np.array([img]))
           prediction = model.predict(img)
           print(f"This digit is probably a {np.argmax(prediction)}")
           plt.imshow(img[0], cmap=plt.cm.binary)
           plt.show()
       except:
           print("Error!")
       finally:
           image_number += 1
   ```

## Results

After training, the model achieves an accuracy of over 97% on the MNIST test dataset. The trained model can accurately predict handwritten digits from new images.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

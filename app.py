import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2
import os

# Set the title of the app
st.title('Image Forgery Detection using ELA')

st.write("""
Upload an image, and the model will predict whether it is **Authentic** or **Fake**.
""")

# Sidebar for examples
st.sidebar.title('ELA Examples')

# Select box to choose example type
example_type = st.sidebar.selectbox('Select example type:', ('Authentic', 'Fake'))

def load_example_images(example_type):
    """
    Loads example images and their corresponding ELA images based on the selected example type.

    Args:
        example_type (str): Type of examples to load ('Authentic' or 'Fake').

    Returns:
        tuple: Two lists containing original images and their ELA images.
    """
    example_dir = os.path.join('examples', example_type.lower())
    images = []
    ela_images = []
    
    # Check if the directory exists
    if not os.path.exists(example_dir):
        st.sidebar.warning(f"The directory '{example_dir}' does not exist.")
        return images, ela_images  # Return empty lists if directory doesn't exist

    # Iterate through files in the directory
    for filename in os.listdir(example_dir):
        # Skip files that are ELA images to avoid duplication
        if '_ela' not in filename and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            img_path = os.path.join(example_dir, filename)
            # Construct the corresponding ELA image filename
            base_name, ext = os.path.splitext(filename)
            ela_img_filename = f"{base_name}_ela.jpg"
            ela_img_path = os.path.join(example_dir, ela_img_filename)
            
            # Check if the ELA image exists
            if os.path.exists(ela_img_path):
                try:
                    img = Image.open(img_path)
                    ela_img = Image.open(ela_img_path)
                    images.append(img)
                    ela_images.append(ela_img)
                except Exception as e:
                    st.sidebar.error(f"Error loading images: {e}")
            else:
                st.sidebar.warning(f"ELA image '{ela_img_filename}' not found for '{filename}'.")
    
    return images, ela_images

# Load example images based on the selected type
images, ela_images = load_example_images(example_type)

# Display example images in the sidebar
st.sidebar.write(f"### {example_type} Examples")
if images:
    for i in range(len(images)):
        st.sidebar.write(f"**Example {i+1}**")
        st.sidebar.image(images[i], caption='Original Image', use_column_width=True)
        st.sidebar.image(ela_images[i], caption='ELA Image', use_column_width=True)
        st.sidebar.write("---")
else:
    st.sidebar.write("No examples available.")

st.sidebar.write("""
**Explanation:**

- **Original Image**: The image before any processing.
- **ELA Image**: The result of Error Level Analysis, highlighting areas of potential manipulation.

**Note**: In fake images, manipulated regions often show higher error levels.
""")

@st.cache_resource
def load_model():
    """
    Loads and compiles the trained TensorFlow model.

    Returns:
        tf.keras.Model: The compiled TensorFlow model.
    """
    try:
        model = tf.keras.models.load_model('detect_fake_img_model.h5', compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.95),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.AUC(name='prc', curve='PR'),
            ]
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the trained model
model = load_model()

def compute_ela_cv(image, quality=95):
    """
    Performs Error Level Analysis (ELA) on the given image.

    Args:
        image (PIL.Image.Image): The original image.
        quality (int, optional): JPEG quality level. Defaults to 95.

    Returns:
        PIL.Image.Image: The ELA processed image.
    """
    temp_filename = 'temp_file_name.jpg'
    SCALE = 15

    # Save the image at a certain quality
    try:
        image.save(temp_filename, 'JPEG', quality=quality)
    except Exception as e:
        st.error(f"Error saving temporary ELA image: {e}")
        return image  # Return the original image if saving fails

    # Open the compressed image and compute the difference
    try:
        compressed_image = Image.open(temp_filename)
    except Exception as e:
        st.error(f"Error opening temporary ELA image: {e}")
        return image  # Return the original image if opening fails

    # Compute the difference
    ela_image = ImageChops.difference(image, compressed_image)

    # Enhance the difference image
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1

    # Scale the difference to enhance features
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image

def preprocess_image(ela_image):
    """
    Preprocesses the ELA image for model prediction.

    Args:
        ela_image (PIL.Image.Image): The ELA processed image.

    Returns:
        np.ndarray: The preprocessed image array.
    """
    try:
        # Resize ELA image to match model's input size
        ela_image = ela_image.resize((224, 224))
        # Convert to NumPy array and normalize
        ela_array = np.array(ela_image).astype('float32') / 255.0
        # Expand dimensions to match the model's input shape
        ela_array = np.expand_dims(ela_array, axis=0)
        return ela_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# File uploader for user images
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'tif'])

if uploaded_file is not None and model is not None:
    try:
        # Open the image file
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
    except Exception as e:
        st.error(f"Error opening uploaded image: {e}")
        st.stop()

    # Perform ELA
    try:
        ela_image = compute_ela_cv(image)
        st.image(ela_image, caption='ELA Image', use_column_width=True)
    except Exception as e:
        st.error(f"Error performing ELA: {e}")
        st.stop()

    # Preprocess ELA image
    processed_image = preprocess_image(ela_image)
    if processed_image is None:
        st.stop()

    # Make prediction
    try:
        prediction = model.predict(processed_image)
        score = prediction[0][0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.stop()

    # Interpret the prediction
    if score > 0.5:
        st.success(f"The image is **Authentic** with a confidence of {score:.2f}")
    else:
        st.error(f"The image is **Fake** with a confidence of {1 - score:.2f}")

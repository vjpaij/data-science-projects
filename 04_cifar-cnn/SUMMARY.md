## Project Overview
This project focuses on building an image classification system using PyTorch, a powerful open-source machine learning framework. The goal is to train a deep learning model capable of accurately identifying and categorizing images of everyday objects. This involves the entire data science lifecycle, from data acquisition and preprocessing to model development, evaluation, and deployment. The project demonstrates proficiency in fundamental deep learning concepts, image processing techniques, and the practical application of PyTorch for a real-world classification task.

## Tools & Technologies
- **Programming Language**: 
    - **Python (version 3.x)** - A versatile language with a rich ecosystem of libraries for data science and machine learning.
- **Deep Learning Framework**: 
    - **PyTorch (version 1.x)** - An open-source machine learning framework known for its flexibility, dynamic computation graphs, and strong GPU acceleration. Excellent for research and development, provides fine-grained control over the model, and has a growing community and production support.
- **Data Manipulation and Analysis**:
    - **Pandas**: For data manipulation and analysis, especially for handling tabular data if you were to analyze metadata associated with images. Provides efficient data structures and tools for cleaning, transforming, and analyzing data.
    - **NumPy**: For numerical computations and array operations, essential for handling image data as multi-dimensional arrays. Fundamental library for numerical computing in Python, offering efficient array operations.
- **Image Processing and Augmentation**:
    - **Pillow (PIL)**: For basic image manipulation tasks like opening, resizing, and converting image formats. A widely used library for image processing in Python.
    - **Torchvision**: A PyTorch library that provides datasets, model architectures, and image transformations specifically designed for computer vision tasks. Simplifies data loading, provides pre-trained models, and offers common image transformations for data augmentation.
- **Visualization**:
    - **Matplotlib**: For creating static, interactive, and animated visualizations in Python, useful for EDA and visualizing model performance. A foundational library for plotting in Python, offering a wide range of customization options.
    - **Seaborn**: Built on top of Matplotlib, providing a higher-level interface for creating informative and attractive statistical graphics. Simplifies the creation of complex and visually appealing statistical plots.
    - **TensorBoard**: A visualization toolkit provided with TensorFlow (and easily integrated with PyTorch) for tracking and visualizing training metrics, model graphs, and more. Crucial for monitoring the training process, debugging model issues, and comparing different experiments.
- **Model Deployment**:
    - **Streamlit**: An open-source Python library for creating interactive web applications for data science and machine learning. Simple and fast way to build user-friendly web interfaces for showcasing your model.
    - **Heroku**: A platform as a service (PaaS) that enables developers to deploy, manage, and scale web applications. Provides a straightforward way to deploy Streamlit applications to the web.
- **Version Control**:
    - **Git**: A distributed version control system for tracking changes in code and collaborating on projects. Essential for managing code, collaborating with others, and tracking project history.
    - **GitHub**: Platforms for hosting Git repositories, facilitating collaboration, and providing project management tools. Provides a central place to store and manage your project code, making it accessible and shareable.
- **Environment Management**:
    - **venv**: Tools for creating isolated Python environments to manage project dependencies. Ensures that your project has the specific library versions it needs without conflicting with other projects.

## Data Dictionary
CIFAR-10 dataset
- 60,000 images (50,000 train + 10,000 test)
- 10 classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- Image size: 32x32 pixels, RGB

| Feature | Description                                                   |
| ------- | ------------------------------------------------------------- |
| `Image` | 32x32 pixel RGB image                                         |
| `Label` | Category of the image (0 to 9)                                |
| `Class` | Airplane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck |

## How to Run
1. Clone this repository
2. Install requirements: python -m venv venv -> source venv/bin/activate -> `pip install -r requirements.txt`
3. Open and run the Jupyter notebook
4. Run streamlit app: streamlit run cnn_app.py

## Additional Resources
- Deep Learning Course: https://www.udemy.com/course/complete-deep-learning-course/
- OpenCv Documentation: https://docs.opencv.org/
- Pytorch Documentatiob: https://www.udemy.com/course/pytorch-for-deep-learning-with-python-bootcamp/
- Streamlit Documentation: https://docs.streamlit.io/
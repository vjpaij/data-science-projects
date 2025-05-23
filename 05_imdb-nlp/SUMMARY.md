## Project Overview
Sentiment analysis is a natural language processing (NLP) technique used to determine the emotional tone behind a piece of text. It's about understanding whether the attitude, emotions, and opinions expressed in text are positive, negative, or neutral. The goal of this project is to build a system that can analyze text data and classify the sentiment expressed in that text. Sentiment analysis is a crucial tool in today's data-driven world because it allows us to process and understand the vast amount of unstructured text data generated online. It helps businesses and organizations make informed decisions based on public opinion.

Here, we are building a high-accuracy sentiment classifier for IMDB movie reviews that:
- Distinguishes between positive (1) and negative (0) reviews
- Handles long-form text (multi-sentence reviews)
- Deploys as a scalable web service

## Tools & Technologies:
- **Python**: The primary programming language.
- **Jupyter Notebook**: For interactive development and documentation.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Scikit-learn**: For machine learning algorithms, model selection, and evaluation.
- **NLTK (Natural Language Toolkit)**: For text preprocessing.
- **PyTorch**: For building and training the deep learning model.
- **Streamlit**: For deploying the model as a web application.
- **Matplotlib and Seaborn**: For data visualization.
- **Regular Expression (re)**: For advanced string processing
- **Transformers**: For using pre-trained models.

## Data Dictionary
- **Dataset**: The project uses the IMDB dataset, which contains a collection of 50,000 movie reviews along with their sentiment labels (positive or negative). The dataset is split into a training set of 25,000 reviews and a test set of 25,000 reviews.
- **Data Fields**:
    - _text_: The text of the movie review. (Feature)
    - _sentiment_: The sentiment of the review, labeled as either 'positive' (1) or 'negative' (0). (Target Variable)
- **Dataset Source**: The IMDB dataset is available from several sources. In this project, we'll use the version available within the Hugging Face datasets library.

## How to Run
1. Clone this repository
2. Install requirements: python -m venv venv -> source venv/bin/activate -> `pip install -r requirements.txt`
3. Open and run the Jupyter notebook
4. Run streamlit app: streamlit run imbd_app.py

## Additional Resources
- Scikit-learn documentation: https://scikit-learn.org/
- NLTK documentation: https://www.nltk.org/
- Streamlit documentation: https://streamlit.io/
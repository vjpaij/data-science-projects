## 1\. Project Overview

 Recommendation systems are fundamental to modern digital platforms, helping users discover products, services, and content tailored to their preferences. We'll build a system that can predict what users might like based on their past behavior and similarities with other users.

**Project Scope**:
*   Implement both collaborative filtering and content-based recommendation approaches
*   Use the MovieLens dataset (a standard benchmark dataset)
*   Cover the full data science pipeline from data loading to deployment
*   Include proper model evaluation and hyperparameter tuning
*   Deploy the final model using Flask for web accessibility  

## 2\. Tools and Technologies

**Core Technologies**:
*   Python 3.9
*   Pandas (data manipulation)
*   NumPy (numerical operations)
*   Scikit-learn (machine learning algorithms)
*   Surprise (recommendation system library)
*   Matplotlib/Seaborn (visualization)
*   Flask (web deployment)

**Development Environment**:
*   Jupyter Notebook (for exploration and prototyping)
*   VS Code(for development)
*   Git/GitHub (version control)
*   Pip (package management)
    
## 3\. Dataset Dictionary

MovieLens dataset (small version with 100,000 ratings) which contains:

**Main Files**:
1.  `ratings.csv`:
    *   `userId` (int): Unique identifier for each user
    *   `movieId` (int): Unique identifier for each movie
    *   `rating` (float): Rating given by the user to the movie (1-5 scale)
    *   `timestamp` (int): Unix timestamp of when the rating was given
        
2.  `movies.csv`:
    *   `movieId` (int): Unique identifier for each movie
    *   `title` (str): Movie title with release year
    *   `genres` (str): Pipe-separated list of genres (e.g., "Action|Adventure|Sci-Fi")
        
3.  `links.csv` (optional for advanced features):
    *   `movieId` (int): Unique identifier for each movie
    *   `imdbId` (str): IMDB identifier
    *   `tmdbId` (str): TMDB identifier
        
**Dataset Statistics**:
*   ~100,000 ratings from 600 users on 9,000 movies
*   Each user has rated at least 20 movies
*   Ratings are on a 5-star scale with half-star increments

## 4\. How to Run the Code

1. Clone this repository
2. Install requirements: python -m venv venv -> source venv/bin/activate -> `pip install -r requirements.txt`
3. Open and run the Jupyter notebook
4. Run flask app: python movie_app.py -> visit http://localhost:5000 in browser

* * *

## 5\. Additional Learning Resources (Udemy Courses)

- "Recommender Systems and Deep Learning in Python" by Lazy Programmer Inc.
- "Machine Learning, Data Science and Deep Learning with Python" by Sundog Education
- "Python for Data Science and Machine Learning Bootcamp" by Jose Portilla

* * *
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from surprise import dump
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

movie_app = Flask(__name__)

# Load the model
model_path = 'models/final_model.pkl'
_, final_model = dump.load(model_path)

# Load movies data
movies = pd.read_pickle('models/movies.pkl')

# Create TF-IDF matrix of movie genres

movies['genres_text'] = movies['genres'].apply(lambda x: ' '.join(x))
tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(' '))
tfidf_matrix = tfidf.fit_transform(movies['genres_text'])

# Compute cosine similarity between movies
content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

def content_based_recommendations(movie_id, n=5):
    # Get index of movie
    idx = movies.index[movies['movieId'] == movie_id].tolist()[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(content_similarity[idx]))
    
    # Sort by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top n similar movies (excluding itself)
    sim_scores = sim_scores[1:n+1]
    
    # Get movie indices and similarity scores
    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    
    # Return recommendations
    return movies.iloc[movie_indices][['movieId', 'title', 'genres']], scores

def hybrid_recommendation(user_id, movie_id=None, n=5):
    if movie_id:
        # Content-based recommendations
        cb_recs, _ = content_based_recommendations(movie_id, n*2)
        
        # Get predicted ratings for these movies from collaborative model
        cb_recs['predicted_rating'] = cb_recs['movieId'].apply(
            lambda x: final_model.predict(user_id, x).est)
        
        # Sort by predicted rating
        return cb_recs.sort_values('predicted_rating', ascending=False).head(n)
    
    else:
        # Get all movies user hasn't rated
        rated_movies = ratings[ratings['userId'] == user_id]['movieId']
        all_movies = set(ratings['movieId'].unique())
        unrated_movies = all_movies - set(rated_movies)
        
        # Get predicted ratings
        predictions = []
        for movie in unrated_movies:
            pred = final_model.predict(user_id, movie)
            predictions.append((movie, pred.est))
        
        # Sort and return top n
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_movies = [x[0] for x in predictions[:n]]
        
        return movies[movies['movieId'].isin(top_movies)]
    
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

@movie_app.route('/')
def home():
    return render_template('index.html')

@movie_app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = int(data['user_id'])
    movie_id = int(data.get('movie_id', 0))
    
    if movie_id:
        # Get hybrid recommendations
        recommendations = hybrid_recommendation(user_id, movie_id)
    else:
        # Get collaborative filtering recommendations
        recommendations = hybrid_recommendation(user_id)
    
    return jsonify(recommendations.to_dict('records'))

if __name__ == '__main__':
    movie_app.run(debug=True)
from email.mime import application  # optional: can be removed if unused
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from bs4 import BeautifulSoup
import pickle
import requests

# Load the trained sentiment classifier and TF-IDF vectorizer
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl', 'rb'))  # ✅ Must be fitted

# Global variables to cache similarity data
data = None
similarity = None

def create_similarity():
    global data, similarity
    data = pd.read_csv('main_data.csv')
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    similarity = cosine_similarity(count_matrix)

def rcmd(m):
    global data, similarity
    m = m.lower()
    if data is None or similarity is None:
        create_similarity()

    if m not in data['movie_title'].unique():
        return 'Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies'
    else:
        i = data.loc[data['movie_title'] == m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)[1:11]
        return [data['movie_title'][x[0]] for x in lst]

def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["', '')
    my_list[-1] = my_list[-1].replace('"]', '')
    return my_list

def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions)

@app.route("/similarity", methods=["POST"])
def similarity_route():
    movie = request.form['name']
    rc = rcmd(movie)
    return rc if isinstance(rc, str) else "---".join(rc)

@app.route("/recommend", methods=["POST"])
def recommend():
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    suggestions = get_suggestions()

    # Convert string lists to actual lists
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)

    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[", "")
    cast_ids[-1] = cast_ids[-1].replace("]", "")

    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"', '\"')

    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}
    cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    print(f"calling imdb api: https://www.imdb.com/title/{imdb_id}/reviews/?ref_=tt_ov_rt")
    url = f'https://www.imdb.com/title/{imdb_id}/reviews/?ref_=tt_ov_rt'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)

    reviews_list, reviews_status = [], []

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'lxml')
        soup_result = soup.find_all("div", {"class": "ipc-html-content-inner-div"})

        for reviews in soup_result:
            if reviews.string:
                reviews_list.append(reviews.string)
                movie_review_list = np.array([reviews.string])
                movie_vector = vectorizer.transform(movie_review_list)  # ✅ Works because vectorizer is fitted
                pred = clf.predict(movie_vector)
                reviews_status.append('Good' if pred else 'Bad')

        movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}

        return render_template('recommend.html', title=title, poster=poster, overview=overview,
                            vote_average=vote_average, vote_count=vote_count,
                            release_date=release_date, runtime=runtime, status=status,
                            genres=genres, movie_cards=movie_cards, reviews=movie_reviews,
                            casts=casts, cast_details=cast_details)
    else:
        print("Failed to retrieve reviews")
        return "Error fetching IMDb reviews"

if __name__ == '__main__':
    app.run(debug=True)
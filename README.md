# MovieLens Recommendation System

Author: Jon McCaffrey

# Contents

### 1. [Overview](#overview)
### 2. [Business and Data Understanding](#bidness)
### 3. [Data Preparation](#dataprep)
### 4. [Modeling](#modeling)
### 5. [Final Model Evaluation](#final)
### 6. [Summary](#summary)

![image1](./images/movie_covers.jpg)

<a id='overview'></a>

# Overview

Using the __["small" dataset](https://grouplens.org/datasets/movielens/latest/)__ from __[MovieLens](https://movielens.org/)__, which contains explicit user ratings, movie titles, genres, etc., we were able to produce a model-based collaborative filtering recommendation system that allows for user input and will return the top 5 recommendations of movies from the dataset that the user has not yet rated.  Our process involved using the Surprise Python scikit library to process data and iterate through multiple recommender algorithms (including SVD and SVD++), using GridSearch, cross-validation and root mean square error (RMSE) to decide on our best-performing model.  Our final model used a tuned SVD algorithm with an RMSE of 0.8457.  We also created a content-based filtering model, using TF-IDF Vectorizer and cosine similarity, to address the "cold start" problem inherent in collaborative filtering models.  With this model, a user who is not already in the dataset can input a known movie from the dataset and be returned 5 movie recommendations to choose from.

<a id='bidness'></a>

# Business and Data Understanding

![image2](./images/just_watch_logo_resize.png)

We will present to the __[JustWatch](https://www.justwatch.com/)__ streaming guide app, with the premise that they can provide movie recommendations to customers who already use their service to search across streaming platforms for a specific movie and create watchlists.  

The MovieLens small dataset contains 100,836 user ratings from 610 different users, each of whom contributed at least 20 ratings.  A total of 9,742 movie titles are included in this dataset.  There are two additional .csv files contained within the package download that we did not end up using.  One file contained 3,683 tags ("a single word or short phrase") that users applied with their numerical rating, which represented less than 4% of the total ratings in the dataset.  The other file contained links of each movie in the dataset to its listing on __[The Movie Database](https://www.themoviedb.org/)__ or on the __[Internet Movie Database](https://www.imdb.com/)__

The MovieLens dataset was specifically developed with recommendation system building in mind.  The data contains explicit user ratings, movie titles, and genres with consistent user ID and movie ID indices across files.  The business problem we were tasked with was to "build a model that provides top 5 movie recommendations to a user, based on their ratings of other movies".  Though movie recommendation systems are now common-place in the streaming apps we have available, understanding the architecture of these everyday models may allow us to expand their use into other realms, or at least refine what a model can currently provide, with the goal of delivering the most refined experience to an individual user.  

Citation included in the README.txt:  
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872>

Additional citations at end of notebook.

<a id='dataprep'></a>

# Data Preparation:

We used Pandas to explore the four different CSV files contained in the small MovieLens.org dataset.  The first file contained user IDs and explicit user ratings.  Each user was selected randomly by MovieLens, and only users with at least 20 ratings were included.  The rating scale runs from 0.5 to 5.0 in half-star increments.   The second file contained movie information, with title and genres.  As stated above, we did not use the remaining two files, one of which contained information for links to TMDB and IMDB, and the last of which included user-generated tags for a small portion (<4%) of movies.  

We combined the movies and user ratings files into one dataframe, using the common 'movieId' column.  This dataset is well-curated, with no nulls to manage in this combined dataframe.  

Next we instantiated Surprise to perform a train-test split (80/20), and started comparing different algorithms

<a id='modeling'></a>

# Modeling

We iterated through many algorithms, using cross-validation and RMSE scores to decide how to optimize our findings. 

The first step, for a baseline comparison, was to use the NormalPredictor algorithm to randomly generate ratings.  This resulted in the highest RMSE of 1.4327.  The next baseline estimated was through BaselineOnly, which utilizes Alternating Least Squares, and the RMSE was substantially better at 0.8683.  We progressed from there to using Singular Value Decomposition, starting with the default hyperparameters (RMSE 0.8698), and then utilizing GridSearchCV to try to optimize hyperparameters.  For the SVD algorithm, this took ~3 mins per GridSearch with the parameters we provided.  We adjusted n_factors (number of user & item factors considered), n_epochs (number of iterations of stochastic gradient descent), lr_all (the learning rate for all parameters), and reg_all (the regularization term for all parameters).  We also consider bias (whether to use baselines) on our first search, but kept that 'True' for the remainder.  Our best parameters yielded a model with an RMSE of 0.8457.  

To be thorough, we also considered Non-negative Matrix Factorization (NMF), as well as the SVD++ model, which is an extension of SVD taking into account implicit ratings.   NMF did not improve our findings (RMSE 0.8668).  Nor did SVD++, even utilizing a time-consuming and computationally expensive GridSearch (best RMSE 0.8545).   Below is a visual comparision of all RMSE

![Comparison of RMSEs from all algorithms tested](./images/suprise_models_bar.png)

<a id='final'></a>

# Final Model Evaluation

To evaluate our final model on the full dataset and actually create a recommendation system, we had additional steps to complete.  First we built our full trainset and fit it our best model on it.  Then we returned to the main Pandas dataframe and created a separate dataframe that contained each movie only once.  Next, we were able to create a separate list for a given user that contained the movies they'd *already* seen, and a separate dataframe of the movies they had **not** yet seen.  We were able to incorporate that information into a function that would return the top 5 movies for any given user that they had not yet seen.  

One limitation of any recommendation system (ours included) is the quality of the user input.  This could include rater bias (always rating highly, or consistently rating every movie at a low score).  It may also be beneficial to have a wider rating scale (e.g. 0-10) to allow for more granularity.  The user-generated tags are also a good idea for additional refinement of the recommendation, but <4% of our dataset contained them.  

Specifically related to model-based collaborative filtering, it is dependent on explicit user ratings, and would be unable to provide recommendations for a new user with no ratings to which to refer.  To be able to offer recommendations even to a new user, we took the opportunity to create a content-based filter, using genre information on each movie in the unique movie dataset.  This involved utilizing sklearn's TF-IDF Vectorizer and cosine similarity.  By finding the top 5 movies with the closest similarity (in genre) to a favorite movie inputted by the user, we were able to provide recommendations even for a user who has no prior movie ratings listed in the dataset.  

<a id='summary'></a>

# Summary

In summary, we were tasked with creating a top 5 movie recommendation system to demonstrate for the JustWatch streaming guide app.  Using the "small" dataset from MovieLens, we were able to produce a model-based collaborative filtering recommendation system that allows for user input and returns the top 5 recommendations of movies from the dataset that the user has not yet rated. We primarily used the Surprise Python scikit library to process data and iterate through multiple recommender algorithms (including SVD and SVD++), using GridSearch, cross-validation and RMSE to decide on our best-performing model. Our final model used a tuned SVD algorithm with an RMSE of 0.8457. We also created a content-based filtering model, using TF-IDF Vectorizer and cosine similarity, to address the "cold start" problem inherent in collaborative filtering models. With this model, a user who is not already in the dataset can input a known movie from the dataset and be returned 5 movie recommendations to choose from.


For More Information
Please review the full analysis in our __[Jupyter Notebook](https://github.com/mccafj/MovieLens-Recommendation-System/blob/main/Final%20Notebook%20MovieLens%20Recommendation%20System.ipynb)__ or the __[PDF of our slide presentation.](https://github.com/mccafj/MovieLens-Recommendation-System/blob/main/Final%20slides.pdf)__

For any additional questions, please contact Jon McCaffrey jonmccaffrey524@gmail.com

Repository Structure

###### ├── __[README.md](https://github.com/mccafj/MovieLens-Recommendation-System#readme)__
###### ├── __[Final MovieLens notebook.ipynb](https://github.com/mccafj/MovieLens-Recommendation-System/blob/main/MovieLens%20notebook.ipynb)__  <- Main notebook for project code
###### ├── __[Final slides.pdf](https://github.com/mccafj/MovieLens-Recommendation-System/blob/main/Final%20slides.pdf)__      <- PDF of slides for our presentation
###### ├── __[Data](https://github.com/mccafj/MovieLens-Recommendation-System/tree/main/data)__			                          <- Location of raw dataset
###### └── __[Images](https://github.com/mccafj/MovieLens-Recommendation-System/tree/main/images)__                              <- images used and generated in the project

---

## Moctale rating patterns and poster analysis

The project now includes a public-page data collector for [Moctale](https://www.moctale.in/) alongside its original MovieLens recommendation analysis. Moctale's visible meter uses **Perfection**, **Go for it**, **Timepass**, and **Skip** categories; these are not converted into assumed 1–5 star values. A small, attributed page snapshot is in `data/moctale_public_snapshot.json`.

Fetch a current public catalog snapshot with a capped, one-second-spaced crawl:

```bash
python moctale_fetch.py --max-pages 20
```

This reads the public homepage and up to 20 linked public content pages, then writes `moctale_public_content.csv`, `moctale_meter_patterns.csv`, `moctale_meter_summary_by_collection.csv`, and `moctale_collections.csv` under `data/moctale_public/`. It records title, collection, visible overview, vibe chart percentages, displayed meter percentages, and poster URL. It does not log in, retrieve private data, copy poster image files, or collect review text/profile information. Missing meter percentages remain blank. The collection summary averages displayed category percentages across pages; it does not infer a vote-weighted score because Moctale does not expose the vote count. Check the site's current terms before using collected material beyond personal analysis.

The checked-in snapshot records what the Avengers: Endgame page displayed: 96% Perfection and 4% Go for it, with no displayed vote count and the page text “No Reviews Yet.” Treat those percentages as a page snapshot, not a statistically supported rating distribution.

The separate MovieLens analysis below is still available for the included ratings dataset. Its findings describe MovieLens, not Moctale.

Install the analysis dependencies and run it from the repository root:

```bash
python -m pip install -r requirements.txt
python moctale_analysis.py
```

The default run reads `data/ratings.csv` and `data/movies.csv`, then writes a report, CSV summaries, and charts to `outputs/moctale/`. Outputs include rating distribution, sentiment-tag counts, monthly rating patterns, genre averages, per-movie vote counts and confidence bands, and per-user rating styles. Movie averages are also Bayesian-shrunk toward the overall mean so low-vote movies are less likely to dominate a ranking; adjust the prior strength with `--prior-votes`.

Monthly patterns use the timestamp when a MovieLens rating was submitted. This is not a movie release or viewing date, so it cannot establish a release-week honeymoon effect. MovieLens also does not provide review text, helpful votes, verified viewing, user trust scores, account age, IP/device data, collection membership, or demographics. The analysis therefore does not estimate those behaviors or apply the speculative trust and recency weights in the pasted Moctale brief.

### Local poster artwork and visual recommendations

The repository does not include movie poster image files. To show artwork in the dashboard or use it for visual similarity, add poster images that you have permission to use to `licensed_posters/`, named with the MovieLens movie ID (for example, `licensed_posters/1.jpg`). Supported formats are JPG, PNG, and WebP. The app serves only images from that local folder; a TMDB ID by itself is not a poster image path.

Install the optional computer-vision dependencies, then run the analysis with the same local folder:

```bash
python -m pip install -r requirements-vision.txt
python moctale_analysis.py --posters licensed_posters --poster-query 1
```

The first run downloads the pretrained TorchVision ResNet-18 weights. It extracts normalized 512-dimensional poster embeddings and caches them under `outputs/moctale/poster_visual_embeddings.npz`; run the analysis again after adding or replacing posters. It also retains the poster dimensions, color/edge indicators, and visual-neighbor CSVs. If PyTorch/TorchVision is not installed, the existing recommender continues to work without visual scores.

Recommendations combine SVD collaborative predictions (50%), genre-profile similarity (30%), and poster-embedding similarity (20%). When a modality is unavailable for a candidate, the available modality weights are renormalized. The “By title” panel also uses a selected MovieLens viewer’s taste profile and the chosen title as a positive seed.

The app’s voting widget stores one adjustable vote per movie and browser visitor ID in the local `instance/moctale_votes.sqlite3` database. Those votes update the displayed meter and visitor preference profile. MovieLens ratings are still the underlying dataset; anonymous browser votes are local to this app instance and do not publish or modify Moctale ratings.

TMDB artwork has not been wired into this ML application. Review the image provider’s current license and API terms before using any third-party catalog images or data for model training, inference, or display. The local poster folder lets you use assets whose applicable rights permit these uses.

### Project scope compared with the Moctale brief

The project now implements data-backed rating summaries, confidence-aware movie averages, user rating-style summaries, local pretrained poster embeddings, a hybrid recommender, and anonymous local Moctale voting. Use cases such as scene recognition, cast identification, trailer tagging, manipulation detection, helpfulness prediction, and platform-specific collection comparisons remain proposed extensions because the source data does not contain the necessary inputs or labels.

### Open-licensed posters (Wikimedia Commons)

```bash
python open_posters.py --popular 2000   # find posters via Wikidata + Commons, keep only open licenses
python fetch_posters.py                 # download them into licensed_posters/
python moctale_analysis.py --posters licensed_posters   # optional: rebuild poster embeddings
```

Only public-domain, CC0, CC BY and CC BY-SA images are kept. Each card shows the poster's author and license with a link to its source. Films without a freely licensed poster get a generated cover.

### Offline data: MovieLens 25M and poster images

```bash
python download_data.py                          # MovieLens 25M -> data/ml-25m/ (262 MB zip, ~1.2 GB unzipped, MD5-checked)
python download_data.py --dataset ml-32m         # or MovieLens 32M (most films, no tag genome)
python download_data.py --posters-only --posters 1000   # lead images of the 1,000 most-rated films -> data/posters/
python analysis.py                               # precompute everything the Analyze tab shows (~1 minute for 25M)
python analysis.py --posters-only                # re-run just the poster analysis after more posters arrive
```

Both downloads resume if interrupted and are git-ignored. Posters come from English Wikipedia, one request at a time with backoff; most are copyrighted and used under fair use, so keep them for private analysis. With posters on disk the site serves them locally and works offline.

`analysis.py` reads the ratings in chunks with compact types (~2 GB RAM for 25M) and writes `outputs/analysis/<dataset>/`:

- **Overview**: rating scale and half-star adoption, activity by year, long tail (Lorenz curve, Gini), viewer activity and generosity, mean rating by release decade, ranked lists (top rated, most divisive, hidden gems…)
- **Films**: a filterable, sortable table of every film plus a popularity-vs-rating scatter
- **Genres**: mean rating, share of ratings, co-occurrence lift heatmap, genre share over decades
- **Latent space**: PureSVD (randomized truncated SVD of the bias-corrected rating matrix, 48 factors), singular-value energy, a t-SNE/PCA map of 3,000 films, nearest neighbours, and how well genre vs. tag-genome similarity agrees with rating-based similarity
- **Viewer segments**: k-means on unit viewer factor vectors, with genre lift, active years, and the films each segment rates higher / watches more than everyone else
- **Tag genome**: browse films by any of 1,128 tags, see a film's signature tags, and compare its nearest films by genome vs. by genre
- **Posters (computer vision)**: brightness, contrast, saturation, colourfulness, edge density, warmth, dark share, tonal entropy, k-means palettes and HSV histograms per poster; genre profiles with confidence intervals, genre palettes, decade trends, correlations with rating/popularity, a poster map, look-alike search, and a cross-validated logistic-regression experiment predicting genre from poster features

### Project report and evaluation

The app's **Data** tab is the ML project report: problem and data, the methods (SVD matrix factorization, user-KNN similarity metrics, genre TF-IDF cosine, the hybrid and its MMR re-rank, Bayesian ranking, poster embeddings), measured results, live model internals, what was built, and limitations.

Every number in it comes from `outputs/report/metrics.json`, produced by:

```bash
python evaluate.py            # ~5 minutes: 5-fold CV, factor and k sweeps, precision/recall@10, hybrid diversity
python evaluate.py --svdpp    # also cross-validate SVD++ (slower)
python evaluate.py --quick    # 3 folds, 20 users; a smoke test
```

The "Inside the model" section calls three endpoints on the running app: `/api/report/explain` (one SVD prediction as μ + b_u + b_i + p_u·q_i), `/api/report/similarity` (genre cosine vs. SVD factor cosine vs. Pearson on co-ratings for two films) and `/api/report/neighbors` (nearest films by learned factors vs. by genre).

### Posters, film details and reviews (no API key)

`wikipedia_source.py` matches each MovieLens film to Wikidata by its IMDb ID and saves into `instance/films.sqlite3`:

- **Wikidata**: English Wikipedia article, runtime, director, Rotten Tomatoes, Metacritic and IMDb scores
- **Wikipedia**: the article's lead image (the poster), its opening paragraph, and the "Critical response" / "Reception" section, stored in the `reviews` table

```bash
python wikipedia_source.py --popular 500     # the 500 most-rated films (a few minutes)
python wikipedia_source.py --all             # every film with an IMDb ID
python wikipedia_source.py --movie-ids 1 318 79132
```

Films that aren't cached are looked up the first time someone opens their page. The film page also shows the free-text tags MovieLens viewers gave each film (`data/tags.csv`).

Wikipedia text is CC BY-SA 4.0 and each film links to its article. Most posters on English Wikipedia are non-free images hosted there under fair use, so check that your use fits before publishing the site.

### Optional: TMDB

TMDB adds backdrops, taglines and user reviews. Copy `.env.example` to `.env`, set `TMDB_READ_TOKEN` (free at https://www.themoviedb.org/settings/api), then run `python tmdb.py --popular 500`. TMDB data merges into the same database and fills gaps without overwriting the Wikipedia fields. This product uses the TMDB API but is not endorsed or certified by TMDB.

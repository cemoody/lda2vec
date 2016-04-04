# Hacker News Comments with lda2vec example
This example trains a multi-component lda2vec model on a corpus of Hacker News
comments. The goal is to model how Hacker News stories have changed in time,
how they correlate with the number of comments posted, and what individual
commenter topics are.

### Running the model

To run this example, first run `preprocess.py` which will download the Hacker
News comments CSV, tokenize it, and quickly build a vocabulary. Once finished,
it saves the training data to file.

Then run `model.py` which will train the lda2vec model. 

Finally, `visualize.py` helps the human interpret what the topics mean.

### The HN Comment Data

The corpus has been slightly filtered. We've removed comments made by 
infrequent users (e.g. having fewer than 10 comments ever) and removed stories
with fewer than 10 comments. The training corpus is available at 
[Zenodo](https://zenodo.org/record/45901#.Vrv5jJMrLMU).

### Preparing the HN Comment Data

You shouldn't need to repeat any of the Google BigQuery work. If you would like
to nevertheless, the rough steps are outline below:

The raw HN data is available on Google BigQuery, see for example these resources:

- Previous analysis on this [dataset](https://github.com/fhoffa/notebooks/blob/master/analyzing%20hacker%20news.ipynb)

- Dataset [shared here](https://bigquery.cloud.google.com/table/fh-bigquery:hackernews.comments)

Data Prepataion

#### Query 1

    SELECT p0.id AS id
         , p0.text as text
         , p0.author AS author
         , p0.ranking AS ranking
         , p0.time
         , p0.time_ts
         , COALESCE(p7.parent, p6.parent, p5.parent, p4.parent, p3.parent, p2.parent, p1.parent, p0.parent) story_id
         , GREATEST(  IF(p7.parent IS null, -1, 7)
                    , IF(p6.parent IS null, -1, 6)
                    , IF(p5.parent IS null, -1, 5)
                    , IF(p4.parent IS null, -1, 4)
                    , IF(p3.parent IS null, -1, 3)
                    , IF(p2.parent IS null, -1, 2)
                    , IF(p1.parent IS null, -1, 1)
                    , 0) level
    FROM    [fh-bigquery:hackernews.comments] p0
    LEFT JOIN EACH [fh-bigquery:hackernews.comments] p1 ON p1.id=p0.parent
    LEFT JOIN EACH [fh-bigquery:hackernews.comments] p2 ON p2.id=p1.parent
    LEFT JOIN EACH [fh-bigquery:hackernews.comments] p3 ON p3.id=p2.parent
    LEFT JOIN EACH [fh-bigquery:hackernews.comments] p4 ON p4.id=p3.parent
    LEFT JOIN EACH [fh-bigquery:hackernews.comments] p5 ON p5.id=p4.parent
    LEFT JOIN EACH [fh-bigquery:hackernews.comments] p6 ON p6.id=p5.parent
    LEFT JOIN EACH [fh-bigquery:hackernews.comments] p7 ON p7.id=p6.parent
    WHERE p0.deleted IS NULL
      AND p0.dead IS NULL
      AND LENGTH(p0.text) > 5
    HAVING level = 0

#### Query 2

    SELECT s.id AS story_id
     , s.time AS story_time
     , s.url AS story_url
     , s.text AS story_text
     , s.author AS story_author
     , c.id AS comment_id
     , c.text AS comment_text
     , c.author AS comment_author
     , c.ranking as comment_ranking
     , author_counts.n_comments AS author_comment_count
     , story_counts.n_comments AS story_comment_count
    FROM [lda2vec-v02:data.comment_to_story_id] c
    JOIN (SELECT story_id
               , COUNT(story_id) AS n_comments
          FROM [lda2vec-v02:data.comment_to_story_id]
          GROUP BY story_id
        ) AS story_counts
    ON c.story_id = story_counts.story_id 
    JOIN (SELECT author
               , COUNT(author) AS n_comments
          FROM [lda2vec-v02:data.comment_to_story_id]
          GROUP BY author
        ) AS author_counts
    ON c.author = author_counts.author 
    JOIN [fh-bigquery:hackernews.stories] s
    ON s.id = c.story_id
    WHERE story_counts.n_comments > 10
      AND author_counts.n_comments > 10

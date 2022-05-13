# BrightFuture

Please look at Final Report Examples for examples on how to use BrightFuture!

The first example shows how BrightFuture loads pre-cleaned datasets, merges them, and builds multiple models, as well as picks the best one, in just 4 lines of code.
The second example demonstrates how in just 2 lines of code, BrightFuture loads a dataset and builds multiple models with them, this time reporting on the best one as we asked for in our `display` argument.

## Example Usage

Clone the repo with `git clone git@github.com:n8kim1/bright-future.git` and then try these in a Python file or Jupyter notebook within `src/`.

Start off with

    import numpy as np
    import pandas as pd
    import json
    from importlib import reload
    import statsmodels.api as sm
    import bright_future_base as bf
    reload(bf)

### Data Loading

Utilise `load_df(dataset, grouped_by)` to load a dataset (try "profs", "awards", or "works") and optionally a `grouped_by` argument (try "author").

### Data Filtering

Utilise `get_works_by_author(author)` to load get an author's publications (try "Samuel Madden").

### Data Aggregation

Utilise `group_works_by_field()` to load get publication counts by field.

### Data Merging

Utilise `merge_datasets(datasets)` to cleanly merge datasets (try `["works", "awards"]`).

### Similarity Metric

Utilise `similarity_by_author(author_1, author_2)`  to get a Cosine Similarity metric for the authors' publications. Try "Samuel Madden" and "Tim Kraska".

### Automated Modeling

Utilise `model_builder(data, responder, predictors, display="best", thresh=1.1)` to automatically build a model and display either "all" models or the "best". Play around with the threshold for how stringent and R2 increase you want in your model.
Example code is as below.

    df_prof = bf.load_df("profs")
    best_model = bf.model_builder(data=df_prof,
                responder="is_uni_top_10",
                predictors=["is_bachelors_top_10", "is_doctorate_top_10"],
                display="all", thresh = 1.0)

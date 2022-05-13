# BrightFuture

Please look at Final Report Examples for examples on how to use BrightFuture!

The first example shows how BrightFuture loads pre-cleaned datasets, merges them, and builds multiple models, as well as picks the best one, in just 4 lines of code.
The second example demonstrates how in just 2 lines of code, BrightFuture loads a dataset and builds multiple models with them, this time reporting on the best one as we asked for in our `display` argument.

## Example Usage

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

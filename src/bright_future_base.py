from sklearn.linear_model import SGDClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd
import json
import statsmodels.api as sm

with open('../data/csrankings/authors-small.json') as f:
    authors_small = json.load(f)

with open('../data/csrankings/authors.json', encoding='UTF-8') as f:
    authors = json.load(f)

with open('../data/csrankings/area-counts-small.json') as f:
    area_counts_small = json.load(f)

with open('../data/profs.tsv') as f:
    profs = pd.read_csv('../data/profs.tsv', sep='\t', header=0)


awards_data_full = pd.read_csv('../data/awards-data.tsv', sep='\t', header=0)


aicolor = "#32CD32"  # limegreen
syscolor = "#00bfff"  # blue
theorycolor = "#ffff00"  # yellow
intercolor = "#ffc0cb"  # pink
nacolor = "#d3d3d3"  # light gray
nocolor = "#ffffff"  # white = no co-authors (making it invisible)
# map area (subarea, publication, etc) to title (field)
areaList = [
    {"area": "ai", "title": "AI", "color": aicolor},
    {"area": "aaai", "title": "AI", "color": aicolor},
    {"area": "ijcai", "title": "AI", "color": aicolor},
    {"area": "vision", "title": "Vision", "color": aicolor},
    {"area": "cvpr", "title": "Vision", "color": aicolor},
    {"area": "iccv", "title": "Vision", "color": aicolor},
    {"area": "eccv", "title": "Vision", "color": aicolor},
    {"area": "mlmining", "title": "ML", "color": aicolor},
    {"area": "kdd", "title": "ML", "color": aicolor},
    {"area": "nips", "title": "ML", "color": aicolor},
    {"area": "icml", "title": "ML", "color": aicolor},
    {"area": "nlp", "title": "NLP", "color": aicolor},
    {"area": "emnlp", "title": "NLP", "color": aicolor},
    {"area": "naacl", "title": "NLP", "color": aicolor},
    {"area": "acl", "title": "NLP", "color": aicolor},
    {"area": "ir", "title": "Web & IR", "color": aicolor},
    {"area": "sigir", "title": "Web & IR", "color": aicolor},
    {"area": "www", "title": "Web & IR", "color": aicolor},
    {"area": "arch", "title": "Arch", "color": syscolor},
    {"area": "asplos", "title": "Arch", "color": syscolor},
    {"area": "hpca", "title": "Arch", "color": syscolor},
    {"area": "isca", "title": "Arch", "color": syscolor},
    {"area": "micro", "title": "Arch", "color": syscolor},
    {"area": "sec", "title": "Security", "color": syscolor},
    {"area": "usenixsec", "title": "Security", "color": syscolor},
    {"area": "oakland", "title": "Security", "color": syscolor},
    {"area": "ccs", "title": "Security", "color": syscolor},
    {"area": "ndss", "title": "Security", "color": syscolor},
    {"area": "pets", "title": "Security", "color": syscolor},
    {"area": "comm", "title": "Networks", "color": syscolor},
    {"area": "sigcomm", "title": "Networks", "color": syscolor},
    {"area": "nsdi", "title": "Networks", "color": syscolor},
    {"area": "mod", "title": "DB", "color": syscolor},
    {"area": "icde", "title": "DB", "color": syscolor},
    {"area": "vldb", "title": "DB", "color": syscolor},
    {"area": "pods", "title": "DB", "color": syscolor},
    {"area": "sigmod", "title": "DB", "color": syscolor},
    {"area": "hpc", "title": "HPC", "color": syscolor},
    {"area": "hpdc", "title": "HPC", "color": syscolor},
    {"area": "ics", "title": "HPC", "color": syscolor},
    {"area": "sc", "title": "HPC", "color": syscolor},
    {"area": "mobile", "title": "Mobile", "color": syscolor},
    {"area": "mobicom", "title": "Mobile", "color": syscolor},
    {"area": "ubicomp", "title": "Mobile", "color": syscolor},
    {"area": "sensys", "title": "Mobile", "color": syscolor},
    {"area": "mobisys", "title": "Mobile", "color": syscolor},
    {"area": "metrics", "title": "Metrics", "color": syscolor},
    {"area": "imc", "title": "Metrics", "color": syscolor},
    {"area": "sigmetrics", "title": "Metrics", "color": syscolor},
    {"area": "ops", "title": "OS", "color": syscolor},
    {"area": "plan", "title": "PL", "color": syscolor},
    {"area": "pacmpl", "title": "PL", "color": syscolor},
    {"area": "popl", "title": "PL", "color": syscolor},
    {"area": "pldi", "title": "PL", "color": syscolor},
    {"area": "oopsla", "title": "PL", "color": syscolor},
    {"area": "icfp", "title": "PL", "color": syscolor},
    {"area": "soft", "title": "SE", "color": syscolor},
    {"area": "issta", "title": "SE", "color": syscolor},
    {"area": "icse", "title": "SE", "color": syscolor},
    {"area": "ase", "title": "SE", "color": syscolor},
    {"area": "fse", "title": "SE", "color": syscolor},
    {"area": "fast", "title": "Systems", "color": syscolor},
    {"area": "usenixatc", "title": "Systems", "color": syscolor},
    {"area": "eurosys", "title": "Systems", "color": syscolor},
    {"area": "sosp", "title": "Systems", "color": syscolor},
    {"area": "osdi", "title": "Systems", "color": syscolor},
    {"area": "act", "title": "Theory", "color": theorycolor},
    {"area": "soda", "title": "Theory", "color": theorycolor},
    {"area": "focs", "title": "Theory", "color": theorycolor},
    {"area": "stoc", "title": "Theory", "color": theorycolor},
    {"area": "crypt", "title": "Crypto", "color": theorycolor},
    {"area": "crypto", "title": "Crypto", "color": theorycolor},
    {"area": "eurocrypt", "title": "Crypto", "color": theorycolor},
    {"area": "log", "title": "Logic", "color": theorycolor},
    {"area": "lics", "title": "Logic", "color": theorycolor},
    {"area": "cav", "title": "Logic", "color": theorycolor},
    {"area": "ec", "title": "Econ", "color": theorycolor},
    {"area": "wine", "title": "Econ", "color": theorycolor},
    {"area": "graph", "title": "Graphics", "color": intercolor},
    {"area": "siggraph", "title": "Graphics", "color": intercolor},
    {"area": "siggraph-asia", "title": "Graphics", "color": intercolor},
    {"area": "chi", "title": "HCI", "color": intercolor},
    {"area": "chiconf", "title": "HCI", "color": intercolor},
    {"area": "uist", "title": "HCI", "color": intercolor},
    {"area": "robotics", "title": "Robotics", "color": intercolor},
    {"area": "rss", "title": "Robotics", "color": intercolor},
    {"area": "iros", "title": "Robotics", "color": intercolor},
    {"area": "icra", "title": "Robotics", "color": intercolor},
    {"area": "bio", "title": "Comp. Biology", "color": intercolor},
    {"area": "ismb", "title": "Comp. Biology", "color": intercolor},
    {"area": "recomb", "title": "Comp. Biology", "color": intercolor},
    {"area": "da", "title": "Design Automation", "color": syscolor},
    {"area": "dac", "title": "Design Automation", "color": syscolor},
    {"area": "iccad", "title": "Design Automation", "color": syscolor},
    {"area": "bed", "title": "Embedded Systems", "color": syscolor},
    {"area": "emsoft", "title": "Embedded Systems", "color": syscolor},
    {"area": "rtas", "title": "Embedded Systems", "color": syscolor},
    {"area": "rtss", "title": "Embedded Systems", "color": syscolor},
    {"area": "vis", "title": "Visualization", "color": intercolor},
    {"area": "vr", "title": "Visualization", "color": intercolor},
    {"area": "na", "title": "Other", "color": nacolor},
    {"area": np.nan, "title": "Other", "color": nacolor},
]
area_dict = {}
for d in areaList:
    a = d['area']
    t = d['title']
    area_dict[a] = t


def load_profs():
    """
    Internal helper function
    """
    return profs


def load_awards():
    """
    Internal helper function
    """
    return awards_data_full


def load_df_works(small=False, grouped_by=None):
    """
    Function to load a DataFrame of publications by author
    Input:
    small = True loads a toy dataset, useful for debugging
    grouped_by states which column name to group by
    Output:
    DataFrame
    """
    if small:
        df_authors = pd.DataFrame.from_dict(authors_small)
    else:
        df_authors = pd.DataFrame.from_dict(authors)

    df_authors['title'] = df_authors['area'].apply(lambda area: area_dict[area])

    # coerce datatypes and interpret numericals correctly
    df_authors['count'] = df_authors['count'].astype(float)
    df_authors['adjustedcount'] = df_authors['adjustedcount'].astype(float)
    # Tried to use int, but NaN exists, so use float
    df_authors['year'] = df_authors['year'].astype(float)
    # for the rest of the strings, commented out for pandas backwards compatibility
    # df_authors = df_authors.convert_dtypes()

    # rename some columns, for clarity
    df_authors = df_authors.rename(columns={"title": "field", "name": "author"})

    if grouped_by == "author":
        df_authors = df_authors.groupby("author").sum().reset_index()
    if grouped_by == "field":
        df_authors = df_authors.groupby("field").sum().reset_index()
    if grouped_by == "area":
        df_authors = df_authors.groupby("area").sum().reset_index()

    return df_authors


def load_award_winners():
    """""
    Toy dataset
    """
    award_winners = {"Kraska Krown": {2021: "Tim Kraska", }, "Madden Memorial (RIP Sam) Award": {2020: "Samuel Madden", },
                     "6.S079 Prof Award": {2021: "Tim Kraska", 2020: "Samuel Madden", }}
    return award_winners


def load_df_awards(grouped_by=None):
    """
    Function to load a DataFrame of Best Paper awards
    Input:
    grouped_by states which column name to group by
    Output:
    DataFrame
    """
    df = pd.read_csv('../data/awards-data.tsv', sep='\t', header=0)
    df = df.rename(columns={"author-name": "author"})
    if grouped_by == "author":
        df = df.assign(award_count=1).groupby(
            "author").sum().drop(columns=["year"]).reset_index()
    if grouped_by == "journal":
        df = df.groupby("journal").sum().reset_index()
    return df


def load_df_profs(grouped_by=None):
    """
    Function to load a DataFrame of Professorship
    Input:
    grouped_by states which column name to group by
    Output:
    DataFrame
    """
    df_prof = pd.read_csv('../data/profs.tsv', sep='\t', header=0)
    uni_rankings = ["Massachusetts Institute of Technology",
                    "Carnegie Mellon University",
                    "Stanford University",
                    "University of California, Berkeley",
                    "University of Illinois at Urbana-Champaign",
                    "Cornell University",
                    "Georgia Institute of Technology",
                    "University of Washington",
                    "Princeton University",
                    "University of Texas at Austin"]

    df_prof["is_uni_top_10"] = df_prof["University"].isin(uni_rankings).astype(int)
    df_prof["is_bachelors_top_10"] = df_prof["Bachelors"].isin(uni_rankings).astype(int)
    df_prof["is_doctorate_top_10"] = df_prof["Doctorate"].isin(uni_rankings).astype(int)
    return df_prof


def load_df_rankings():
    """
    Function to load a DataFrame of Top 10 US Universities in CS
    Output:
    DataFrame
    """
    df = pd.read_csv('../data/uni_rankings.tsv', sep='\t', header=None)
    df = df.rename(columns={0: "Rank", 1: "University"})
    return df


def load_df(dataset, grouped_by=None):
    """
    Function to load a dataset as asked for by the user
    Input:
    grouped_by states which column name to group by
    Output:
    DataFrame
    """
    if dataset == "profs":
        return load_df_profs(grouped_by=grouped_by)
    elif dataset == "awards":
        return load_df_awards(grouped_by=grouped_by)
    elif dataset == "works":
        return load_df_works(small=False, grouped_by=grouped_by)
    elif dataset == "uni_rankings":
        return load_df_rankings()

# FILTERS


def filter_works(df_works=None, author=None, year=None):
    """
    Filter based on given arguments
    Input:
    df_works
    author
    year
    Output:
    Filtered DataFrame
    """
    if df_works is None:
        df_works = load_df("works")
    if author is not None:
        df_works = df_works[(df_works['author'] == author)]
    if year is not None:
        df_works = df_works[(df_works['year'] == year)]
    return df_works


def get_works_by_author(author, df_works=None):
    """
    Filter based on given arguments
    Input:
    author
    df_works
    year
    Output:
    Filtered DataFrame
    """
    if df_works is None:
        df_works = load_df("works")
    return filter_works(df_works, author=author, year=None)

# AGGREGATORS


def group_works_by_field(df_works=None):
    """
    Group based on given arguments
    Input:
    df_works
    Output:
    Grouped DataFrame, summed
    """
    if df_works is None:
        df_works = load_df("works")
    return df_works.groupby('field').sum()

# TODO option for raw count or adjusted count


def count_works_by_field(df_works=None):
    """
    Count works by each field
    Input:
    df_works
    Output:
    Dictionary of grouped DataFrame, summed
    """
    if df_works is None:
        df_works = load_df("works")
    grouped = group_works_by_field(df_works)
    return grouped['count'].to_dict()


def count_works_by_field_filter_name(df_works, author):
    """
    Count works by each field and author
    Input:
    df_works
    author
    Output:
    Dictionary of grouped DataFrame, summed
    """
    return count_works_by_field(get_works_by_author(df_works, author))


# MERGER

def merge_datasets(datasets):
    """
    Merge any two datasets
    Input:
    List of two datasets
    Output:
    Merged DataFrame
    """
    if sorted(datasets) == ["awards", "works"]:
        df_works_author = load_df("works", grouped_by="author")
        df_awards_author = load_df("awards", grouped_by="author")
        merged_df = df_works_author.merge(df_awards_author, on="author", how="left")
        merged_df = merged_df.fillna(0).drop(columns=["year"])
        return merged_df
    if sorted(datasets) == ["profs", "works"]:
        df_prof = load_df("profs")
        df_works = load_df("works")
        merged = df_works.merge(df_prof, left_on='author', right_on='Fullname', how='inner')
        merged = merged.drop(columns=["Fullname"])
        return merged
    if sorted(datasets) == ["awards", "profs"]:
        df_profs = load_df("profs")
        df_awards = load_df("awards")
        merged = df_awards.merge(df_profs, left_on='author', right_on='Fullname', how='inner')
        merged = merged.drop(columns=["Fullname"])
        return merged

# SIMILARITY STUFF

# Group works by feild, (optionally filter years), then run similarity on the counts by field.
# Compute similarity by area or field? should use option


def similarity_works(df_works_1, df_works_2, metric='cosine', adjusted=False):
    """
    Internal Helper Function
    """
    # note title -> field, bad naming
    works_by_title_1 = count_works_by_field(df_works_1)
    works_by_title_2 = count_works_by_field(df_works_2)
    v = DictVectorizer()
    works_stacked = [works_by_title_1, works_by_title_2]
    works_encoded = v.fit_transform(works_stacked)
    # print(works_by_title_1, works_by_title_2)
    return cosine_similarity(works_encoded)[0][1]

# A wrapper around similarity_works, when you're only needing authors


def similarity_by_author(author_1, author_2, df_works=None, year=None, metric='cosine', adjusted=False):
    """
    Get cosine similarity metric for two authors
    Input:
    author_1
    author_2
    df_works
    year
    metric = 'cosine'
    adjusted = False
    Output:
    Cosine Similarity between the two authors' works in the given year (optionally)
    """
    if not df_works:
        df_works = load_df("works")
    df_works_1 = filter_works(df_works, author=author_1, year=year)
    df_works_2 = filter_works(df_works, author=author_2, year=year)
    return similarity_works(df_works_1, df_works_2, metric=metric, adjusted=adjusted)


# MACHINE LEARNING

def model_builder(data, responder, predictors, display="best", thresh=1.1):
    """
    Automatically build models and choose best one
    Input:
    data
    responder
    predictors
    display = 'best' or 'all'
    thresh
    Output:
    Modeling results
    """
    features_choices = predictors

    features_chosen = []
    features_prospective = []

    res_best = None
    r2_best_chosen = 0
    res_best_prospective = None
    r2_best_prospective = 0
    features_best_prospective = None

    # Iteratively build a model. (stop if the model already has all the possible features)
    while len(features_chosen) < len(features_choices):
        # Given the existing model, we test adding each new future, see if there's improvement, and what the best is.

        # First, test adding each new feature to the existing model
        r2_best_prospective = 0
        for feature in features_choices:
            # Only add _new_ features
            if feature not in features_chosen:
                # build features from the old set, add one
                features_prospective = features_chosen.copy()
                features_prospective.append(feature)
                mod = sm.OLS(data[responder], sm.add_constant(data[features_prospective]))
                res = mod.fit()

                if res.rsquared > r2_best_prospective:
                    r2_best_prospective = res.rsquared
                    features_best_prospective = features_prospective
                    res_best_prospective = res

        # now we have the best prospective model, and its r2.
        # if this represents a 10% bump in r2 then use it

        if display == 'all':
            # display the best generated model for any round of iteration,
            # even if its not the best across all rounds.
            print("Iteration's best model Features:", features_best_prospective)
            print("Iteration's best model r2:", r2_best_prospective)
            # print("Iteration's best model summary:")
            # print(res_best_prospective.summary())

        if r2_best_prospective >= thresh * r2_best_chosen:
            # use it! track its score too
            r2_best_chosen = r2_best_prospective
            features_chosen = features_best_prospective
            res_best = res_best_prospective
        else:
            # not a sufficient boost. terminate with the previous best; this round of search produced nothing
            break

    print("Overall Best model Features:", features_chosen)
    print("Overall Best model summary:")
    print(res_best.summary())

    return res_best


# Predictor

def train_classifier(df_works, train_data_X, train_data_y):
    """
    Internal toy function
    """
    w1 = count_works_by_field_filter_name(df_works, 'Tim Kraska')
    w2 = count_works_by_field_filter_name(df_works, 'Samuel Madden')
    w3 = count_works_by_field_filter_name(df_works, 'David R. Karger')

    works_stacked = [w1, w2, w3]
    v = DictVectorizer()
    works_encoded = v.fit_transform(works_stacked)

    lin_sgd = SGDClassifier(loss='log')
    lin_sgd.fit(works_encoded[0:3], [0, 1, 0])

    return lin_sgd


def predict_classifier(classifier, predict_data_X):
    """
    Internal toy function
    """
    w1 = count_works_by_field('Tim Kraska')
    w2 = count_works_by_field('Samuel Madden')
    w3 = count_works_by_field('David R. Karger')

    works_stacked = [w1, w2, w3]
    v = DictVectorizer()
    works_encoded = v.fit_transform(works_stacked)

    return classifier.predict_proba(works_encoded[0])

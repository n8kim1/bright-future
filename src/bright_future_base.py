import numpy as np
import pandas as pd
import json
with open('../data/csrankings/authors-small.json') as f:
    authors = json.load(f)

with open('../data/csrankings/area-counts-small.json') as f:
    area_counts = json.load(f)

a = 33

def load_df_authors():
    df_authors = pd.DataFrame.from_dict(authors)
    aicolor = "#32CD32"  # limegreen
    syscolor = "#00bfff"  # blue
    theorycolor = "#ffff00"  # yellow
    intercolor = "#ffc0cb"  # pink
    nacolor = "#d3d3d3"  # light gray
    nocolor = "#ffffff"  # white = no co-authors (making it invisible)

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
    ]
    area_dict = {}
    for d in areaList:
        a = d['area']
        t = d['title']
        area_dict[a] = t
    df_authors['title'] = df_authors['area'].apply(lambda area: area_dict[area])

    # coerce datatypes
    # interpret numerical correctly
    df_authors['count'] = df_authors['count'].astype(float)
    df_authors['adjustedcount'] = df_authors['adjustedcount'].astype(float)
    df_authors['year'] = df_authors['year'].astype(int)
    # for the rest of the strings
    df_authors = df_authors.convert_dtypes()


    return df_authors

def load_award_winners():
    award_winners = {"Kraska Krown": {2021: "Tim Kraska",} , "Madden Memorial (RIP Sam) Award": {2020: "Samuel Madden",} ,
    "6.S079 Prof Award": {2021: "Tim Kraska", 2020: "Samuel Madden",}}
    return award_winners
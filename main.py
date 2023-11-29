# %%
from typing import Dict, List, Set
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

# %% [markdown]
# # Netflix Movies and Shows:
# ## Decoding Trends in Entertainment: A Deep Dive into Netflix's Content Library
# %%
df_netflix = pd.read_csv(
    "./imdb_movies_shows.csv", converters={"production_countries": pd.eval}
).sort_values("release_year")

# %%
df_netflix.head()
# %%
px.histogram(df_netflix, x="type", color="type", labels={"type": "Type"})

# %%
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df_netflix["release_year"].unique(),
        y=df_netflix.groupby("release_year").count()["imdb_score"],
        name="Movies + Shows",
        mode="lines+markers",
    )
)

fig.add_trace(
    go.Scatter(
        x=df_netflix.query("type == 'MOVIE'")["release_year"].unique(),
        y=df_netflix.groupby(["release_year", "type"])
        .count()
        .reset_index()
        .query("type == 'MOVIE'")["imdb_score"],
        name="Movies",
        mode="lines+markers",
    )
)

fig.add_trace(
    go.Scatter(
        x=df_netflix.query("type == 'SHOW'")["release_year"].unique(),
        y=df_netflix.groupby(["release_year", "type"])
        .count()
        .reset_index()
        .query("type == 'SHOW'")["imdb_score"],
        name="Shows",
        mode="lines+markers",
    )
)

fig.update_layout(
    title="Shows and movies released over the years",
    xaxis_title="Release Year",
    yaxis_title="Count",
    legend_title="Legend",
)
fig.show()

# %% [markdown]
# 2019 is the year with the most **movies and shows** released, and it is the same
# year with the most **movies** released. However, for **shows** it falls just
# short of 2018 by 8 shows released.
# %% [markdown]
# # Separating Movies and Shows into two different dataframes

# %%
df_shows = df_netflix.loc[df_netflix["type"] == "SHOW"].drop(columns="type")
df_movies = df_netflix.loc[df_netflix["type"] == "MOVIE"].drop(columns="type")

# %% [markdown]
# ## Distribution of the age rating of Shows and Movies in 2019
# %%
px.histogram(
    df_movies.loc[df_movies["release_year"] == 2019],
    x="age_certification",
    category_orders={"age_certification": ["G", "PG", "PG-13", "R", "NC-17"]},
    title="Movies in 2019 by age certification",
    labels={"age_certification": "age certification"},
    color="age_certification",
)
# %%
px.histogram(
    df_shows.loc[df_shows["release_year"] == 2019],
    x="age_certification",
    category_orders={
        "age_certification": ["TV-Y", "TV-Y7", "TV-G", "TV-PG", "TV-14", "TV-MA"]
    },
    color="age_certification",
    title="Shows in 2019 by age certification",
    labels={"age_certification": "age certification"},
)
# %%
fig = go.Figure()

x = df_movies.dropna(subset="imdb_score")["release_year"].unique()
df_movies_year_rating = df_movies.dropna(subset="imdb_score").groupby("release_year")[
    "imdb_score"
]

fig.add_trace(
    go.Scatter(
        x=x,
        y=df_movies_year_rating.max(),
        mode="lines+markers",
        name="Maximum IMDB rating",
    )
)

y_err = df_movies.groupby("release_year")["imdb_score"].sem().round(2)
fig.add_trace(
    go.Scatter(
        x=x,
        y=df_movies_year_rating.mean().round(2),
        mode="lines+markers",
        name="Average IMDB rating",
        error_y=dict(
            type="data",
            array=y_err,
            visible=True,
        ),
        hovertemplate="Year: %{x}"
        + "<br>Rating: %{y:.2f} +- %{customdata[0]:.2f}"
        + "<br>Count: %{customdata[1]}",
        customdata=np.stack((y_err, df_movies_year_rating.count().values), axis=-1),
    )
)

fig.add_trace(
    go.Scatter(
        x=x,
        y=df_movies_year_rating.min(),
        mode="lines+markers",
        name="Minimum IMDB rating",
    )
)

fig.update_yaxes(range=[0, 10])

fig.update_layout(
    title="Movie ratings across the years",
    xaxis_title="Release Year",
    yaxis_title="Rating",
    legend_title="Legend",
)

# %%
fig = go.Figure()

x = df_shows.dropna(subset="imdb_score")["release_year"].unique()

df_shows_year_rating = df_shows.dropna(subset="imdb_score").groupby("release_year")[
    "imdb_score"
]
fig.add_trace(
    go.Scatter(
        x=x,
        y=df_shows_year_rating.max(),
        mode="lines+markers",
        name="Maximum IMDB rating",
        hovertemplate="Year: %{x}" + "<br> Max: %{y}",
    )
)

y_err = df_shows_year_rating.sem().round(2)
fig.add_trace(
    go.Scatter(
        x=x,
        y=df_shows_year_rating.mean().round(2),
        mode="lines+markers",
        name="Average IMDB rating",
        error_y=dict(
            type="data",
            array=y_err,
            visible=True,
        ),
        hovertemplate="Year: %{x}"
        + "<br>Rating: %{y:.2f} +- %{customdata[0]:.2f}"
        + "<br>Count: %{customdata[1]}",
        customdata=np.stack((y_err, df_shows_year_rating.count().values), axis=-1),
    )
)

fig.add_trace(
    go.Scatter(
        x=x,
        y=df_shows_year_rating.min(),
        mode="lines+markers",
        name="Minimum IMDB rating",
        hovertemplate="Year: %{x}" + "<br> Min: %{y}",
    )
)

fig.update_yaxes(range=[0, 10])

fig.update_layout(
    title="Show ratings across the years",
    xaxis_title="Release Year",
    yaxis_title="Rating",
    legend_title="Legend",
)

# %% [markdown]
# It seems that from 2000 onwards we see a strong decrease in the 'worst'
# rating given to productions across both movies and shows.

df_netflix
# %%
px.histogram(x=df_netflix["production_countries"].str.len())
# %% [markdown]
# Entries with 0 production countries are removed as they are probably
# faulty data.
# %%
df_production = df_netflix.loc[df_netflix["production_countries"].str.len() > 0]
# %%

# Separate by country production number and count productions per country.
fig = px.histogram(
    df_production,
    x=df_production.production_countries.str.len(),
    title="Number of Production Countries per Entry",
    color="type",
    barmode="group",
    histnorm="percent",
)

fig.update_layout(
    xaxis=dict(
        tickmode="linear", tick0=1, dtick=1, title="Number of Production Countries"
    )
)
# %% [markdown]
# Almost all shows, ca. 96%, are produced in 1 country. Wherease for movies
# there is a sizeable amount, ca. 15%, that is produced in more than
# one country.

# %%

df_production["production_countries"]

# %%

# get a set of unique countries

set_of_countries = set()
df_production["production_countries"].apply(lambda x: set_of_countries.update(x))

# Check for integrity of all country ISO codes
for x in set_of_countries:
    if len(x) > 2:
        print(x)
# %% [markdown]
# It seems that Lebanon has been labeled twice, once in its full name and one in
# its ISO country code 'LB'.
# %%

# Find and change that to LB
df_production.loc[
    df_production["production_countries"].apply(lambda x: "Lebanon" in x)
].index

# %%
df_production.loc[649, "production_countries"] = ["LB"]

# %%
set_of_countries = set()
df_production["production_countries"].apply(lambda x: set_of_countries.update(x))
len(set_of_countries)
# %% [markdown]
# There are 106 countries in which production took place.

# %%


def productions_per_country(series: pd.Series, set_of_countries: Set) -> pd.DataFrame:
    p_per_c = dict(zip(set_of_countries, np.zeros(len(set_of_countries))))
    for list_of_countries in series:
        for country in list_of_countries:
            p_per_c[country] += 1

    return pd.DataFrame(dict(country=p_per_c.keys(), n_productions=p_per_c.values()))


# %%
df_productions_per_country = productions_per_country(
    series=df_production["production_countries"],
    set_of_countries=set_of_countries,
)
df_productions_per_country.head()
# %%
fig = px.bar(
    df_productions_per_country.sort_values("n_productions", ascending=False),
    x="country",
    y="n_productions",
    log_y=True,
    hover_name="country",
    hover_data={"country": False, "n_productions": True},
    width=1800,
)

fig.update_layout(
    title="Number of Productions per Country",
    xaxis_title="Country (ISO Code)",
    yaxis_title="Number of Productions; log scale",
)
fig.show()

# %%

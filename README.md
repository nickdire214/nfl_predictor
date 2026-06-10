# NFL Predictor

A pipeline for predicting NFL player prop outcomes (e.g., yards, receptions,
touchdowns) by ingesting player statistics, sportsbook odds, injury reports,
weather data, and snap counts, transforming them into model-ready features,
and training/evaluating prediction models against those props.

## Setup (Windows)

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Then fill in your actual API keys in `.env`.

## Note on the virtual environment

The `venv/` directory is not portable between machines. If you transfer this
project to a different machine, delete any existing `venv/` and repeat the
setup steps above to create a fresh virtual environment.

# Rappi ML Challenge

This repository contains a solution for the Rappi ML Challenge.

# Layout

The project is composed of the following folders:
- api: Flask API that returns a prediction for a titanic passenger payload.
- artifacts: Objects generated during the training process like one hot encoders, models, constants. They are persisted as pickle files.
- data: Initial Titanic dataset. It is used as the initial input of the pipeline.
- pipeline: Package in charge of reading the initial Titanic dataset and producing models ready to use.

# Requirements

- Docker version: 19.03.11
- Docker compose version: 1.24.1

# Installation

- Run the following docker commands

`docker-compose build`

`docker-compose up -d`

- After running the previous commands the api container will be available at port 5000. You can test it as follows

```
curl --location --request POST 'localhost:5000/predict' \
--header 'Content-Type: application/json' \
--data-raw '{
    "PassengerId": 1,
    "Pclass": 3,
    "Name": "Braund, Mr. Owen Harris",
    "Sex": "male",
    "Age": 22.0,
    "SibSp": 1,
    "Parch": 0,
    "Ticket": "A/5 21171",
    "Fare": 7.25,
    "Cabin": null,
    "Embarked": "S"
}'
```

- The response will look like this:

```
{"data":{"linsvc_prediction":0,"rf_prediction":0}}
```

# TODOS

- Implement pipeline profiling.
- Add API profiling.
- Add API proper error handling
- Add API monitoring and alerts
- Add A/B testing.

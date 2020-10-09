class PayloadValidator:
    def __init__(self, payload):
        self.payload = payload
        self.error_message = None

    def is_valid(self):
        required_columns = set(
            [
                "PassengerId",
                "Pclass",
                "Name",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Ticket",
                "Fare",
                "Cabin",
                "Embarked",
            ]
        )

        given_columns = set(self.payload.keys())
        missing_columns = required_columns - given_columns
        invalid_columns = given_columns - required_columns

        if missing_columns or invalid_columns:
            error_message = ""
            if missing_columns:
                error_message = (
                    error_message + f"Missing columns: {list(missing_columns)}."
                )
            if invalid_columns:
                error_message = (
                    error_message + f"Invalid columns: {list(invalid_columns)}."
                )
            self.error_message = error_message
            return False
        return True

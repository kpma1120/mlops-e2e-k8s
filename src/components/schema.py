from numpy import float64, int64
from pandera.pandas import Check, Column, DataFrameSchema


def allow_dtypes(*allowed):
    return Check(lambda s: str(s.dtype) in allowed)

# Schema used for validation of raw data
raw_schema = DataFrameSchema({
    "PassengerId": Column(
        None, nullable=False,
        checks=[
            allow_dtypes("int8","int32","int64"), 
            Check.in_range(min_value=1, max_value=10000)
        ]
    ),
    "Survived": Column(
        None, nullable=False,
        checks=[allow_dtypes("int8","int32","int64"), Check.isin([0,1])]
    ),
    "Pclass": Column(
        None, nullable=False,
        checks=[allow_dtypes("int8","int32","int64"), Check.isin([1,2,3])]
    ),
    "Name": Column(None, nullable=False, checks=allow_dtypes("object","string")),
    "Sex": Column(
        None, nullable=False, 
        checks=[allow_dtypes("object","string"), Check.isin(["male","female"])]
    ),
    "Age": Column(
        None, nullable=True,
        checks=[
            allow_dtypes("float32","float64"), 
            Check.in_range(min_value=0, max_value=120)
        ]
    ),
    "SibSp": Column(
        None, nullable=False,
        checks=[
            allow_dtypes("int8","int32","int64"), 
            Check.in_range(min_value=0, max_value=10)
        ]
    ),
    "Parch": Column(
        None, nullable=False,
        checks=[
            allow_dtypes("int8","int32","int64"), 
            Check.in_range(min_value=0, max_value=10)
        ]
    ),
    "Ticket": Column(None, nullable=False, checks=allow_dtypes("object","string")),
    "Fare": Column(
        None, nullable=False,
        checks=[
            allow_dtypes("float32","float64"), 
            Check.in_range(min_value=0, max_value=600)
        ]
    ),
    "Cabin": Column(None, nullable=True, checks=allow_dtypes("object","string")),
    "Embarked": Column(
        None, nullable=True,
        checks=[allow_dtypes("object","string"), Check.isin(["C","Q","S"])]
    ),
})

# Schema used for validation of preprocessed data, processed data, and retrieved data 
# from Redis feature store
processed_schema = DataFrameSchema({
    "PassengerId": Column(
        int64, nullable=False,
        checks=Check.in_range(min_value=1, max_value=10000)
    ),
    "Survived": Column(
        int64, nullable=False,
        checks=Check.isin([0, 1])
    ),
    "Age": Column(
        float64, nullable=False,
        checks=Check.in_range(min_value=0, max_value=120)
    ),
    "Fare": Column(
        float64, nullable=False,
        checks=Check.in_range(min_value=0, max_value=600)
    ),
    "Pclass": Column(
        float64, nullable=False,
        checks=Check.isin([1, 2, 3])
    ),

    # OneHot Sex
    "Sex_0": Column(float64, nullable=False, checks=Check.isin([0, 1])),
    "Sex_1": Column(float64, nullable=False, checks=Check.isin([0, 1])),

    # OneHot Embarked
    "Embarked_0": Column(float64, nullable=False, checks=Check.isin([0, 1])),
    "Embarked_1": Column(float64, nullable=False, checks=Check.isin([0, 1])),
    "Embarked_2": Column(float64, nullable=False, checks=Check.isin([0, 1])),

    # Family features
    "FamilySize": Column(
        float64, nullable=False,
        checks=Check.in_range(min_value=0, max_value=15)
    ),
    "IsAlone": Column(float64, nullable=False, checks=Check.isin([0, 1])),

    # Cabin flag
    "HasCabin": Column(float64, nullable=False, checks=Check.isin([0, 1])),

    # OneHot Title
    "Title_0": Column(float64, nullable=False, checks=Check.isin([0, 1])),
    "Title_1": Column(float64, nullable=False, checks=Check.isin([0, 1])),
    "Title_2": Column(float64, nullable=False, checks=Check.isin([0, 1])),
    "Title_3": Column(float64, nullable=False, checks=Check.isin([0, 1])),
    "Title_4": Column(float64, nullable=False, checks=Check.isin([0, 1])),

    # Interaction features
    "Pclass_Fare": Column(
        float64, nullable=False,
        checks=Check.in_range(min_value=0, max_value=1800)
    ),
    "Age_Fare": Column(
        float64, nullable=False,
        checks=Check.in_range(min_value=0, max_value=72000)
    ),
})

# Schema used for validation of input raw data to the predict API endpoint for inference
inference_schema = DataFrameSchema({
    "PassengerId": Column(
        int64, nullable=False,
        checks=Check.in_range(min_value=1, max_value=10000)
    ),
    "Pclass": Column(
        float64, nullable=False,
        checks=Check.isin([1, 2, 3])
    ),
    "Name": Column(
        str, nullable=False
    ),
    "Sex": Column(
        str, nullable=False,
        checks=Check.isin(["male", "female"])
    ),
    "Age": Column(
        float64, nullable=True,
        checks=Check.in_range(min_value=0, max_value=120)
    ),
    "SibSp": Column(
        float64, nullable=False,
        checks=Check.in_range(min_value=0, max_value=10)
    ),
    "Parch": Column(
        float64, nullable=False,
        checks=Check.in_range(min_value=0, max_value=10)
    ),
    "Ticket": Column(
        str, nullable=False
    ),
    "Fare": Column(
        float64, nullable=False,
        checks=Check.in_range(min_value=0, max_value=600)
    ),
    "Cabin": Column(
        str, nullable=True
    ),
    "Embarked": Column(
        str, nullable=True,
        checks=Check.isin(["C", "Q", "S"])
    ),
}, coerce=True)

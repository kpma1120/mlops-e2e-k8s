import sys

import pytest
from airflow.models import DagBag

# Generate `dag_ids` during the collection phase to avoid fixture/skip confusion
dag_ids = []
if not sys.platform.startswith("win"):
    dag_ids = list(DagBag(dag_folder="dags", include_examples=False).dags.keys())


@pytest.fixture(scope="module")
def dag_bag() -> DagBag:
    """Fixture: Load all DAGs once per module."""
    if sys.platform.startswith("win"):
        pytest.skip("Airflow not supported on Windows")
    bag = DagBag(dag_folder="dags", include_examples=False)
    assert len(bag.dags) > 0, f"No DAGs loaded: {bag.import_errors}"
    return bag


def test_no_import_errors(dag_bag: DagBag) -> None:
    """Ensure all DAGs can be imported without errors."""
    assert len(dag_bag.import_errors) == 0, (
        f"DAG import failures: {dag_bag.import_errors}"
    )
    

@pytest.mark.parametrize("dag_id", dag_ids)
def test_dag_structure(dag_bag: DagBag, dag_id: str) -> None:
    """Ensure each DAG has at least one task."""
    dag = dag_bag.dags[dag_id]
    assert len(dag.tasks) > 0, f"{dag_id} has no tasks defined"


@pytest.mark.parametrize("dag_id", dag_ids)
def test_dag_retries(dag_bag: DagBag, dag_id: str) -> None:
    """Ensure each DAG has retries >= 2 if defined."""
    dag = dag_bag.dags[dag_id]
    retries = dag.default_args.get("retries", None)
    if retries is not None:
        assert retries >= 2, f"{dag_id} must have retries >= 2"

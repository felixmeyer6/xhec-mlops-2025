"""Serve weekly training and hourly batch prediction with Prefect 2.13.7."""

import argparse
import asyncio

from prefect import serve
from prefect.client.orchestration import get_client
from prefect.client.schemas.schedules import CronSchedule
from prefect.deployments.runner import RunnerDeployment
from prefect.exceptions import ObjectNotFound
from prefect.utilities.asyncutils import sync_compatible

from lib.config import ARTIFACTS_FILEPATH, INPUT_FILEPATH, PREDICTIONS_FILEPATH, TEST_FILEPATH, TRAIN_FILEPATH
from lib.workflows import batch_predict_workflow, train_model_workflow


@sync_compatible
async def build_deployments() -> tuple[RunnerDeployment, RunnerDeployment]:
    """Build schedules without starting a server, creating registrations or blocking."""
    training = await train_model_workflow.to_deployment(
        name="weekly-training",
        version="1.0.0",
        tags=["taxi", "training"],
        schedule=CronSchedule(cron="0 0 * * 1", timezone="UTC"),
        parameters={"train_filepath": TRAIN_FILEPATH, "test_filepath": TEST_FILEPATH, "artifacts_filepath": ARTIFACTS_FILEPATH},
    )
    prediction = await batch_predict_workflow.to_deployment(
        name="hourly-prediction",
        version="1.0.0",
        tags=["taxi", "inference"],
        schedule=CronSchedule(cron="0 * * * *", timezone="UTC"),
        parameters={
            "input_filepath": INPUT_FILEPATH,
            "artifacts_filepath": ARTIFACTS_FILEPATH,
            "output_filepath": PREDICTIONS_FILEPATH,
        },
    )
    return training, prediction


async def serve_deployments() -> None:
    """Serve both flows until interrupted; cleanup runs in a fresh event loop."""
    deployments = await build_deployments()
    await serve(*deployments, pause_on_shutdown=False)


async def pause_deployments() -> None:
    """Pause schedules outside the cancelled runner loop, while the API is reachable."""
    deployments = await build_deployments()
    async with get_client() as client:
        for deployment in deployments:
            try:
                registered = await client.read_deployment_by_name(f"{deployment.flow_name}/{deployment.name}")
            except ObjectNotFound:
                continue
            await client.update_schedule(registered.id, active=False)
            print(f"Paused {deployment.flow_name}/{deployment.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Print schedules without registering or serving them.")
    arguments = parser.parse_args()
    if arguments.check:
        for deployment in build_deployments():
            print(f"{deployment.name}: {deployment.schedule.json()}")
    else:
        try:
            asyncio.run(serve_deployments())
        except KeyboardInterrupt:
            print("Runner stopped.")
        finally:
            # Prefect 2.13 can cancel its own cleanup during Ctrl+C. A fresh loop
            # ensures the pause requests are completed before the script exits.
            asyncio.run(pause_deployments())

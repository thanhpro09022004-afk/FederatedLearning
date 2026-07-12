from typing import Iterable, Optional
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAdagrad
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
import pickle
from dataclasses import asdict



class CustomFedAdagrad(FedAdagrad):
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training and maybe do LR decay."""
        # Decrease learning rate by a factor of 0.5 every 5 rounds
        if server_round % 5 == 0 and server_round > 0:
            config["lr"] *= 0.5
            print("LR decreased to:", config["lr"])
        # Pass the updated config and the rest of arguments to the parent class
        return super().configure_train(server_round, arrays, config, grid)


    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""

        # Convert replies to a list before iterating over them so the parent
        # strategy can still aggregate the same replies afterwards.
        replies = list(replies)
        for reply in replies:
            if reply.has_content():
                # Retrieve the ConfigRecord from the message
                config_record = reply.content["train_metadata"]
                metadata_bytes = config_record["meta"]
                # Deserialize it
                train_meta = pickle.loads(metadata_bytes)
                print(asdict(train_meta))
        # Aggregate the ArrayRecords and MetricRecords as usual
        return super().aggregate_train(server_round, replies)

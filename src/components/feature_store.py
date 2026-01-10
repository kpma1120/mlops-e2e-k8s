import json
from collections.abc import Hashable, Sequence
from typing import Any, cast

import redis


class RedisFeatureStore:
    """Redis-based feature store for managing ML features."""
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0) -> None:
        """Initialize Redis client connection with host, port, and db."""
        self.client: redis.StrictRedis = redis.StrictRedis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )

    def reset(self) -> None:
        """Clear all features stored in Redis under the feature store namespace."""
        keys: list[str] = cast(list[str], self.client.keys("entity:*:features"))
        if keys:
            self.client.delete(*keys)

    def store_features(self, entity_id: int | str, features: dict[str, Any]) -> None:
        """Store features for a single entity."""
        key: str = f"entity:{entity_id}:features"
        self.client.set(key, json.dumps(features))

    def get_features(self, entity_id: int | str) -> dict[str, Any] | None:
        """Retrieve features for a single entity."""
        key: str = f"entity:{entity_id}:features"
        features: str | None = cast(str | None, self.client.get(key))
        if features:
            return json.loads(features)
        return None

    def store_batch_features(
            self, 
            batch_data: dict[Hashable, dict[Hashable, Any]]
        ) -> None:
        """Store features for a batch of entities."""
        for entity_id, features in batch_data.items():
            self.store_features(entity_id, features)

    def get_batch_features(
            self, 
            entity_ids: Sequence[int | str]
        ) -> dict[int | str, dict[str, Any] | None]:
        """Retrieve features for a batch of entities."""
        batch_features: dict[int | str, dict[str, Any] | None] = {}
        for entity_id in entity_ids:
            batch_features[entity_id] = self.get_features(entity_id)
        return batch_features

    def get_all_entity_ids(self) -> list[str]:
        """Retrieve all entity IDs currently stored in Redis."""
        keys: list[str] = cast(list[str], self.client.keys("entity:*:features"))
        entity_ids: list[str] = [key.split(":")[1] for key in keys]
        return entity_ids

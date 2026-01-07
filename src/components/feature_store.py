import redis
import json
from typing import Any, Dict, Hashable, List, Sequence, cast


class RedisFeatureStore:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0) -> None:
        self.client: redis.StrictRedis = redis.StrictRedis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )

    def reset(self) -> None:
        """Clear all features stored in Redis under the feature store namespace."""
        keys: List[str] = cast(List[str], self.client.keys("entity:*:features"))
        if keys:
            self.client.delete(*keys)

    def store_features(self, entity_id: int | str, features: Dict[str, Any]) -> None:
        """Store features for a single entity."""
        key: str = f"entity:{entity_id}:features"
        self.client.set(key, json.dumps(features))

    def get_features(self, entity_id: int | str) -> Dict[str, Any] | None:
        """Retrieve features for a single entity."""
        key: str = f"entity:{entity_id}:features"
        features: str | None = cast(str | None, self.client.get(key))
        if features:
            return json.loads(features)
        return None

    def store_batch_features(self, batch_data: Dict[Hashable, Dict[Hashable, Any]]) -> None:
        """Store features for a batch of entities."""
        for entity_id, features in batch_data.items():
            self.store_features(entity_id, features)

    def get_batch_features(self, entity_ids: Sequence[int | str]) -> Dict[int | str, Dict[str, Any] | None]:
        """Retrieve features for a batch of entities."""
        batch_features: Dict[int | str, Dict[str, Any] | None] = {}
        for entity_id in entity_ids:
            batch_features[entity_id] = self.get_features(entity_id)
        return batch_features

    def get_all_entity_ids(self) -> List[str]:
        """Retrieve all entity IDs currently stored in Redis."""
        keys: List[str] = cast(List[str], self.client.keys("entity:*:features"))
        entity_ids: List[str] = [key.split(":")[1] for key in keys]
        return entity_ids

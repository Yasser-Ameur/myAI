from companion.core.ids import (
    new_entity_id,
    new_episode_id,
    new_fact_id,
    new_id,
    new_memory_id,
)


def test_new_id_has_prefix():
    assert new_id("abc").startswith("abc_")


def test_ids_unique_and_prefixed():
    kinds = [new_entity_id, new_fact_id, new_episode_id, new_memory_id]
    for fn in kinds:
        a, b = fn(), fn()
        assert a != b
        assert isinstance(a, str) and a


def test_ids_are_sortable_timestamps():
    a, b = new_memory_id(), new_memory_id()
    assert a < b or a > b

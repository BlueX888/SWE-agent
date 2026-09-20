import copy
import json
from pathlib import Path

import pytest

from sweagent.agent.history_processors import LastNObservations, TagToolCallObservations
from sweagent.types import History


def get_history(traj_path: Path):
    return json.loads((traj_path).read_text())["history"]


def count_elided_observations(history: History):
    return len([entry for entry in history if "Old environment output" in entry["content"]])


@pytest.fixture
def test_history(test_trajectories_path: Path):
    return get_history(
        test_trajectories_path
        / "gpt4__swe-agent-test-repo__default_from_url__t-0.00__p-0.95__c-3.00__install-1/6e44b9__sweagenttestrepo-1c2844.traj"
    )


def test_last_n_observations(test_history: History):
    processor = LastNObservations(n=3)
    new_history = processor(test_history)
    total_observations = len([entry for entry in test_history if entry["message_type"] == "observation"])
    # extra -1 because instance template is kept
    expected_elided_observations = total_observations - 3 - 1
    assert count_elided_observations(new_history) == expected_elided_observations


def test_add_tag_to_edits(test_history: History):
    processor = TagToolCallObservations(tags={"test"}, function_names={"edit"})
    new_history = processor(test_history)
    for i, entry in enumerate(new_history):
        if entry.get("action", "").startswith("edit "):  # type: ignore
            assert new_history[i + 1].get("tags") == ["test"], entry


def test_keep_output_tag_keeps_old_observation(test_history: History):
    """A keep_output tag on a matching tool call keeps its observation from being elided."""
    tagged = TagToolCallObservations(tags={"keep_output"}, function_names={"open"})(copy.deepcopy(test_history))
    processed = LastNObservations(n=1)(tagged)
    open_obs_idx = next(i for i, e in enumerate(test_history) if e.get("action", "").startswith("open "))
    assert "Old environment output" not in processed[open_obs_idx + 1]["content"]


def test_remove_output_tag_elides_matching_observation(test_history: History):
    """A remove_output tag elides the matching observation without crashing on the action."""
    tagged = TagToolCallObservations(tags={"remove_output"}, function_names={"find_file"})(copy.deepcopy(test_history))
    processed = LastNObservations(n=1)(tagged)
    find_obs_idx = next(i for i, e in enumerate(test_history) if e.get("action", "").startswith("find_file "))
    assert "Old environment output" in processed[find_obs_idx + 1]["content"]

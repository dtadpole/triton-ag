"""
Conversation selection helper for tree-based multi-turn generation.
Selects a conversation from the previous turn based on different strategies.
Uses file-based coordination to ensure all generations in a turn use the same selected conversation.
"""

import asyncio
import json
import os
import random
import re

from logger import logger


async def select_conversation_from_previous_turn(
    turn_id: int,
    record_folder: str,
    min_num_generations: int = 3,
    selection_strategy: str = "random",
    timeout: int = 300,
):
    """
    Select a conversation from the previous turn.
    Uses file-based coordination to ensure all generations in the same turn
    use the same selected conversation (true tree structure).

    Args:
        turn_id: Current turn ID (0-indexed)
        record_folder: Directory containing conversation files
        min_num_generations: Minimum number of generations to wait for
        selection_strategy: Strategy for selection ("random", "best_speedup")
        timeout: Maximum time to wait in seconds

    Returns:
        Dictionary with selected conversation data, or None for turn 0 or on error
    """
    # Turn 0: No selection needed
    if turn_id == 0:
        return {"turn_id": turn_id, "message": "Turn 0 - no selection needed"}

    # For turn > 0, perform selection with robust retry logic
    previous_turn = turn_id - 1
    selection_file = (
        f"{record_folder}/t{turn_id:02d}_selected_from_t{previous_turn:02d}.json"
    )
    os.makedirs(record_folder, exist_ok=True)

    # Retry loop: keep trying until we get a valid selection or timeout
    max_retries = 3
    retry_count = 0
    start_time = asyncio.get_event_loop().time()

    while asyncio.get_event_loop().time() - start_time < timeout:
        # Check if selection already exists
        if os.path.exists(selection_file):
            try:
                await asyncio.sleep(0.1)  # Brief pause to ensure file is fully written
                with open(selection_file, "r") as f:
                    selected = json.load(f)

                # Load the full conversation data
                result = await _load_conversation_data(selected)
                if result is not None:
                    return result
                else:
                    logger.warning(
                        f"⚠️ [select_conversation] Failed to load data from selection file"
                    )
                    await asyncio.sleep(1)
                    continue
            except Exception as e:
                logger.warning(
                    f"⚠️ [select_conversation] Error reading selection file: {e}, retrying..."
                )
                await asyncio.sleep(1)
                continue

        # No selection file exists, try to be the one to create it
        if retry_count >= max_retries:
            logger.error(
                f"❌ [select_conversation] Max retries ({max_retries}) reached"
            )
            await asyncio.sleep(
                2
            )  # Wait a bit before checking if another worker succeeded
            retry_count = 0
            continue

        retry_count += 1

        try:
            # Perform selection
            result = await _perform_selection(
                turn_id,
                previous_turn,
                record_folder,
                min_num_generations,
                selection_strategy,
                timeout
                - (asyncio.get_event_loop().time() - start_time),  # Remaining timeout
            )

            if result is None:
                logger.warning(
                    f"⚠️ [select_conversation] Selection returned None, retrying..."
                )
                await asyncio.sleep(1)
                continue

            # Prepare selection data
            selection_data = {
                "gen_tag": result["gen_tag"],
                "turn_tag": result["turn_tag"],
                "conversation_path": f"{record_folder}/{result['turn_tag']}_conversation.json",
                "completion_path": f"{record_folder}/{result['turn_tag']}_completion.json",
                "generated_code_path": f"{record_folder}/{result['turn_tag']}_generated_code.py",
                "generated_eval_path": f"{record_folder}/{result['turn_tag']}_generated_eval.json",
            }

            # CRITICAL: Verify all files exist before writing selection file
            # This ensures we never create an invalid selection
            missing_files = []
            for key, path in selection_data.items():
                if key.endswith("_path") and not os.path.exists(path):
                    missing_files.append(path)

            if missing_files:
                logger.error(
                    f"❌ [select_conversation] Turn {turn_id} - Files missing from selected result: {missing_files}. "
                    f"This should not happen - bug in _perform_selection!"
                )
                await asyncio.sleep(2)
                continue

            # Write selection data to temp file first
            temp_file = f"{selection_file}.tmp.{os.getpid()}"
            try:
                with open(temp_file, "w") as f:
                    json.dump(selection_data, f, indent=2)

                # Atomic file creation using os.link() - fails if destination exists
                # This is the ONLY way to atomically create without overwriting
                try:
                    os.link(temp_file, selection_file)
                    # Success! We created the selection file
                    logger.info(
                        f"💾 Turn {turn_id} - Created selection: {result['turn_tag']}"
                    )
                    os.remove(temp_file)  # Clean up temp file
                    return result
                except FileExistsError:
                    # Another worker created it first - use theirs instead
                    os.remove(temp_file)
                    await asyncio.sleep(0.1)
                    continue  # Loop back to read the existing file

            except Exception as e:
                logger.warning(
                    f"⚠️ [select_conversation] Error creating selection file: {e}"
                )
                try:
                    os.remove(temp_file)
                except:
                    pass
                await asyncio.sleep(0.5)
                continue

        except Exception as e:
            logger.error(f"❌ [select_conversation] Error during selection: {e}")
            await asyncio.sleep(1)
            continue

    logger.error(
        f"❌ [select_conversation] Timeout after {timeout}s waiting for valid selection"
    )
    return None


async def _perform_selection(
    turn_id: int,
    previous_turn: int,
    record_folder: str,
    min_num_generations: int,
    selection_strategy: str,
    timeout: int,
):
    """Perform the actual selection logic."""
    start_time = asyncio.get_event_loop().time()
    available_generations = []

    while asyncio.get_event_loop().time() - start_time < timeout:
        available_generations = []

        # Scan for available conversations from previous turn
        if os.path.exists(record_folder):
            all_files = os.listdir(record_folder)

            for filename in all_files:
                # Pattern: gen_XX_tYY_conversation.json where YY == previous_turn
                pattern = rf"^gen_(\d+)_t{previous_turn:02d}_conversation\.json$"
                match = re.match(pattern, filename)
                if match:
                    gen_idx = int(match.group(1))
                    found_gen_tag = f"gen_{gen_idx:02d}"
                    found_turn_tag = f"{found_gen_tag}_t{previous_turn:02d}"

                    # Check if all required files exist
                    conversation_path = (
                        f"{record_folder}/{found_turn_tag}_conversation.json"
                    )
                    completion_path = (
                        f"{record_folder}/{found_turn_tag}_completion.json"
                    )
                    generated_code_path = (
                        f"{record_folder}/{found_turn_tag}_generated_code.py"
                    )
                    generated_eval_path = (
                        f"{record_folder}/{found_turn_tag}_generated_eval.json"
                    )

                    all_exist = all(
                        os.path.exists(p)
                        for p in [
                            conversation_path,
                            completion_path,
                            generated_code_path,
                            generated_eval_path,
                        ]
                    )

                    if all_exist:
                        available_generations.append(
                            {
                                "gen_tag": found_gen_tag,
                                "turn_tag": found_turn_tag,
                                "conversation_path": conversation_path,
                                "completion_path": completion_path,
                                "generated_code_path": generated_code_path,
                                "generated_eval_path": generated_eval_path,
                            }
                        )
        else:
            logger.warning(
                f"[_perform_selection] Record folder does not exist: {record_folder}"
            )

        if len(available_generations) >= min_num_generations:
            break

        await asyncio.sleep(1)

    # Check if we got enough generations
    if len(available_generations) < min_num_generations:
        logger.error(
            f"❌ Turn {turn_id} - Timeout waiting for {min_num_generations} "
            f"generations from turn {previous_turn}. Found {len(available_generations)}"
        )
        if available_generations:
            logger.error(f"Available: {[g['turn_tag'] for g in available_generations]}")
        return None

    logger.info(
        f"🎯 Turn {turn_id} - Selecting from {len(available_generations)} generations (turn {previous_turn})"
    )

    # Select based on strategy
    if selection_strategy == "random":
        selected = random.choice(available_generations)
    elif selection_strategy == "best_speedup":
        # Load all evaluations and select the one with best speedup
        best_gen = None
        best_speedup = -1
        for gen in available_generations:
            try:
                with open(gen["generated_eval_path"], "r") as f:
                    eval_data = json.load(f)
                    speedup = eval_data.get("speedup", 0)
                    if speedup > best_speedup:
                        best_speedup = speedup
                        best_gen = gen
            except Exception as e:
                logger.warning(
                    f"⚠️ [select_conversation] Error reading eval for "
                    f"{gen['gen_tag']}: {e}"
                )
                continue
        selected = (
            best_gen if best_gen is not None else random.choice(available_generations)
        )
    else:
        logger.warning(
            f"⚠️ [select_conversation] Unknown selection strategy: "
            f"{selection_strategy}, using random"
        )
        selected = random.choice(available_generations)

    # Load the selected conversation data
    result = await _load_conversation_data(selected)

    if result is not None:
        logger.info(
            f"✅ Turn {turn_id} - Selected {selected['turn_tag']} (strategy: {selection_strategy})"
        )

    return result


async def _load_conversation_data(selected):
    """Load the full conversation data from file paths."""
    try:
        with open(selected["conversation_path"], "r") as f:
            conversation = json.load(f)
        with open(selected["completion_path"], "r") as f:
            completion = json.load(f)
        with open(selected["generated_code_path"], "r") as f:
            generated_code = f.read()
        with open(selected["generated_eval_path"], "r") as f:
            generated_eval = json.load(f)

        return {
            "gen_tag": selected["gen_tag"],
            "turn_tag": selected["turn_tag"],
            "conversation": conversation,
            "completion": completion,
            "generated_code": generated_code,
            "generated_eval": generated_eval,
        }
    except Exception as e:
        logger.error(
            f"❌ [_load_conversation_data] Error loading conversation data: {e}"
        )
        return None


async def main():
    """Test the conversation selector."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--turn_id", type=int, default=1)
    parser.add_argument("--record_folder", type=str, required=True)
    parser.add_argument("--min_num_generations", type=int, default=1)
    parser.add_argument("--selection_strategy", type=str, default="random")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    result = await select_conversation_from_previous_turn(
        turn_id=args.turn_id,
        record_folder=args.record_folder,
        min_num_generations=args.min_num_generations,
        selection_strategy=args.selection_strategy,
        timeout=args.timeout,
    )

    logger.info(f"Selection result: {result}")


if __name__ == "__main__":
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    asyncio.run(main())

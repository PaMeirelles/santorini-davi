import numpy as np
from game.board import Board
from database.models import God


def board_to_tensor(board: Board, phase_idx: int = 0) -> np.ndarray:
    """
    Transforms a Board object into a (31, 5, 5) numpy array.
    Perspective: Current player (the one whose turn it is).

    Args:
        board: The current Board state.
        phase_idx: 0 for Move, 1 for Build 1, 2 for Build 2.
    """
    # 1. Initialize tensor (Expanded to 31 planes)
    # Planes 0-27: Existing state
    # Planes 28-30: Phase indicator (one-hot)
    tensor = np.zeros((31, 5, 5), dtype=np.float32)

    # 2. Map Heights (Planes 0-3)
    for i in range(25):
        row, col = divmod(i, 5)
        height = board.blocks[i]
        if 1 <= height <= 4:
            tensor[height - 1, row, col] = 1.0

    # 3. Identify Active vs Waiting Player
    if board.turn == 1:  # Gray's turn
        my_workers = board.workers[0:2]
        opp_workers = board.workers[2:4]
        my_god_idx = board.gods[0].value
        opp_god_idx = board.gods[1].value
        is_player_1 = 1.0
    else:  # Blue's turn
        my_workers = board.workers[2:4]
        opp_workers = board.workers[0:2]
        my_god_idx = board.gods[1].value
        opp_god_idx = board.gods[0].value
        is_player_1 = 0.0

    # 4. Fill Worker Planes (4-5)
    for idx in my_workers:
        tensor[4, idx // 5, idx % 5] = 1.0
    for idx in opp_workers:
        tensor[5, idx // 5, idx % 5] = 1.0

    # 5. Fill Context Planes (6-7)
    tensor[6].fill(is_player_1)  # Turn indicator
    if board.prevent_up_next_turn:
        tensor[7].fill(1.0)  # Athena effect

    # 6. Fill God Identity Planes (8-27)
    tensor[8 + my_god_idx].fill(1.0)  # My God
    tensor[18 + opp_god_idx].fill(1.0)  # Opponent God

    # 7. Fill Phase Planes (28-30) - NEW
    # This allows the model to differentiate its policy output for the same board
    if 0 <= phase_idx <= 2:
        tensor[28 + phase_idx].fill(1.0)

    return tensor
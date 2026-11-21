"""Full-position equivalence under the eight symmetries of the 5x5 board."""
from functools import lru_cache


def _mapping(transpose, flip_x, flip_y):
    # Map destination squares to their source squares.
    result = [None] * 25
    for source in range(25):
        x, y = source % 5, source // 5
        if transpose:
            x, y = y, x
        if flip_x:
            x = 4 - x
        if flip_y:
            y = 4 - y
        result[5*y+x] = source
    return tuple(result)


TRANSFORMS = tuple(_mapping(t, x, y) for t in (False, True)
                   for x in (False, True) for y in (False, True))
assert len(set(TRANSFORMS)) == 8
POLICY = 'D4: all rotations/reflections/diagonal transpositions; full state and ordered player roles preserved.'


def validate_position(position):
    if (len(position) != 54 or any(position[i] not in '01234' for i in range(0, 50, 2))
        or any(position[i] not in 'NGB' for i in range(1, 50, 2))
        or position[50] not in '01' or not position[51:53].isascii()
        or not position[51:53].isdigit() or position[53] not in '01'):
        raise ValueError('Expected a 54-character engine board with complete state metadata.')


def transform_position(position, transform):
    validate_position(position)
    return ''.join(position[2*i:2*i+2] for i in TRANSFORMS[transform]) + position[50:]


@lru_cache(maxsize=200000)
def canonical_board(cells):
    if len(cells) != 50:
        raise ValueError('Expected 25 height/occupant pairs.')
    return min(''.join(cells[2*i:2*i+2] for i in mapping) for mapping in TRANSFORMS)


@lru_cache(maxsize=400000)
def canonical_key(position):
    validate_position(position)
    return canonical_board(position[:50]) + position[50:]


def distinct_ranked_positions(ordered):
    """Keep the first supplied representative of each full-position class."""
    seen = set()
    for item in ordered:
        key = canonical_key(item['position'])
        if key not in seen:
            seen.add(key)
            yield dict(item, symmetry_key=key)

"""Handle trajectory-end atom ordering using input geometry, never labels."""

import numpy as np
from iso17_data import Frames, make_topology


def adapt_trajectory_ends(frames, tops, block_size):
    molecules = frames.molecules.copy()
    adapted = dict(tops)
    changes = []
    for index in range(block_size-1, len(frames), block_size):
        old = tops[int(molecules[index])]
        actual = make_topology(frames.numbers[index], frames.position[index], valence_cap=True)
        number_changed = int(np.count_nonzero(old.numbers != actual.numbers))
        bond_changed = int(np.count_nonzero(old.bond_edges != actual.bond_edges))
        if number_changed or bond_changed:
            key = 10000 + index
            adapted[key] = actual
            changes.append({"official_frame_id":index+1,"bucket_molecule_id":int(molecules[index]),
                            "number_slots_changed":number_changed,"directed_bond_slots_changed":bond_changed})
            molecules[index] = key
    return Frames(frames.position,frames.force,frames.numbers,molecules,frames.ids), adapted, changes

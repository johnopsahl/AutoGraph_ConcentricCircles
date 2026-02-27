import map_bitmap_to_spiral as sprl
import numpy as np

def test_calc_spiral_arc_length():
    assert sprl.calc_spiral_arc_length(np.pi, 1.5) == 9.164879000938939

def test_find_theta_from_length():
    assert sprl.find_theta_from_length(1, 1.5, 0) == 0.6276102477078693
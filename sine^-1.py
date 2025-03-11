# Generate sine^-1 lookup table
# Save it to sine-1.csv
# This avoids expensive computation of sine^-1

import numpy as np
SAMPLE_ANGLE_DIFF = 0.001  # in radians
SAMPLE_MIN = -1 * np.pi    # in radians
SAMPLE_MAX = 1 * np.pi     # in radians

# Generate lookup table
input_angles = np.arange(SAMPLE_MIN, SAMPLE_MAX, SAMPLE_ANGLE_DIFF)
lookup_table = np.sin(input_angles)

# Save it to sine^-1.csv: first column is input_angles, second column is lookup_table
np.savetxt('sine^-1.csv', np.column_stack((input_angles, lookup_table)), delimiter=',')

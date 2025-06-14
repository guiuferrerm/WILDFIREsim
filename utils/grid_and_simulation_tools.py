import numpy as np

def generate_kernel_and_vectors(radius_m, cellSizeX, cellSizeY):
    max_offset_y = int(np.ceil(radius_m / cellSizeY))
    max_offset_x = int(np.ceil(radius_m / cellSizeX))

    kernel_size_y = 2 * max_offset_y + 1
    kernel_size_x = 2 * max_offset_x + 1
    kernel_array = np.zeros((kernel_size_y, kernel_size_x), dtype=float)
    vectors = []

    for dy in range(-max_offset_y, max_offset_y + 1):
        for dx in range(-max_offset_x, max_offset_x + 1):
            real_dx = dx * cellSizeX
            real_dy = dy * cellSizeY
            distance = np.sqrt(real_dx**2 + real_dy**2)

            factor = 0.25

            if distance <= radius_m:
                i = dy + max_offset_y
                j = dx + max_offset_x

                # Center cell gets weight of 1.0 exactly
                if dx == 0 and dy == 0:
                    kernel_array[i, j] = 1.0
                else:
                    kernel_array[i, j] = 1.0 / (factor*distance + 1e-8)

                if not (dx == 0 and dy == 0):
                    vectors.append([dy, dx])

    vectors = np.array(vectors, dtype=int).reshape(-1, 2)
    
    return kernel_array, vectors

def calculate_delta_height_array_with_vector(heightInitialArray, vector):
    heightArray = np.copy(heightInitialArray)

    if vector[0] > 0:
        heightArray[-vector[0]:, :] = heightArray[0,:].reshape(1, -1)
    elif vector[0] < 0:
        heightArray[:-vector[0], :] = heightArray[-1,:].reshape(1, -1)
    if vector[1] > 0:
        heightArray[:, -vector[1]:] = heightArray[:,0].reshape(-1, 1)
    elif vector[1] < 0:
        heightArray[:, :-vector[1]] = heightArray[:,-1].reshape(-1, 1)
    
    shifted_height = np.roll(heightArray, (vector[0], vector[1]), axis=(0,1))
    deltaHeight = heightInitialArray - shifted_height

    return deltaHeight

def shift_array_and_fill_with_value(array, vector, value):
    if vector[0] > 0:
        array[-vector[0]:, :] = value
    elif vector[0] < 0:
        array[:-vector[0], :] = value
    if vector[1] > 0:
        array[:, -vector[1]:] = value
    elif vector[1] < 0:
        array[:, :-vector[1]] = value
    
    shifted_array = np.roll(array, (vector[0], vector[1]), axis=(0, 1))
    return shifted_array
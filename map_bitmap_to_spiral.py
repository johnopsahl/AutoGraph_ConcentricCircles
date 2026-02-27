import numpy as np
from scipy.optimize import root
from scipy.spatial import KDTree
import svgwrite
from PIL import Image
from pathlib import Path

def calc_spiral_arc_length(theta, b):
    """Calculates arc length L of spiral r = a * theta from 0 to theta."""
    if theta == 0:
        return 0.0
    # Exact formula for Archimedean spiral arc length
    return (b / 2.0) * (theta * np.sqrt(1 + theta**2) + np.log(theta + np.sqrt(1 + theta**2)))

def find_theta_from_length(target_length, b, initial_theta_guess):
    """Numerically solves for theta given a target arc length."""
    # Define the function whose root we want to find: f(theta) - target_L = 0
    func = lambda t: calc_spiral_arc_length(t, b) - target_length

    # root() expects the function to accept and return array-like values
    result = root(func, initial_theta_guess)

    if not result.success:
        raise RuntimeError(f"Root finding failed: {result.message}")

    return result.x[0]

def generate_spiral_segment_points(a, b, segment_length, theta_start, theta_end):
    """Generates line segment points along an Archimedes spiral."""

    # Create an array of theta values from 0 to theta_end with a step size determined by the desired segment length
    theta = []
    theta_current = theta_start

    while theta_current < theta_end:
        length_initial = calc_spiral_arc_length(theta_current, b)
        length_target = length_initial + segment_length
        theta_new = find_theta_from_length(length_target, b, theta_current)
        theta.append(theta_new)
        theta_current = theta_new
    theta = np.array(theta)

    # Calculate the radius for each theta using the formula r = a + b*theta
    r = a + b * theta
    
    # Convert polar coordinates (r, theta) to Cartesian coordinates (x, y)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return np.column_stack((x,y))

def create_segments_from_segment_points(seg_points):
    '''Creates line segments from an array of segment points. Each segment is defined by two consecutive points.'''
    return np.lib.stride_tricks.sliding_window_view(seg_points, 2, axis=0).transpose(0, 2, 1)

def remove_out_of_bounds_segments(segment, x_min, x_max, y_min, y_max):
    '''Removes line segments that are out of bounds'''
    x_coords = segment[:, :, 0]  # shape (N, 2)
    y_coords = segment[:, :, 1]  # shape (N, 2)

    mask = (
        (x_coords >= x_min).all(axis=1) &
        (x_coords <= x_max).all(axis=1) &
        (y_coords >= y_min).all(axis=1) &
        (y_coords <= y_max).all(axis=1)
    )

    return segment[mask]

def nearest_neighbor_line_sort(lines):
    '''Sorts line segments in a nearest neighbor order, starting from the first segment. Returns the order of indices and the sorted lines.'''
    lines = np.array(lines)
    n = len(lines)

    # Build a single static KDTree over all 2n endpoints
    # Layout: point 2i = start of line i, point 2i+1 = end of line i
    points = lines.reshape(-1, 2)  # shape (2n, 2)
    tree = KDTree(points)

    visited = np.zeros(n, dtype=bool)
    order = []
    flipped = []

    current_pos = lines[0][1]  # exit end of first line
    visited[0] = True
    order.append(0)
    flipped.append(False)

    for _ in range(n - 1):
        k = 2
        while True:
            dists, idxs = tree.query(current_pos, k=min(k, 2 * n))
            for idx in idxs:
                line_i = idx // 2
                if not visited[line_i]:
                    is_flipped = (idx % 2 == 1)
                    visited[line_i] = True
                    order.append(line_i)
                    flipped.append(is_flipped)
                    current_pos = lines[line_i][0] if is_flipped else lines[line_i][1]
                    break
            else:
                k *= 2
                continue
            break

    # Build sorted lines, reversing any that were flipped
    flipped = np.array(flipped)
    sorted_lines = lines[order]
    sorted_lines[flipped] = sorted_lines[flipped, ::-1]

    return order, sorted_lines

def write_segments_to_svg(svg_file_path, segment, segement_color, view_box):
    '''Writes line segments to an svg file with color based on segment color.'''

    dwg = svgwrite.Drawing(str(svg_file_path) + "_spiral.svg")

    for i, seg in enumerate(segment):          

        dwg.add(dwg.line(seg[0], seg[1], stroke=svgwrite.rgb(segement_color[i], segement_color[i], segement_color[i]), stroke_width=0.1))

    dwg.viewbox(minx=view_box[0], miny=view_box[1], 
                width=view_box[2], height=view_box[3])
    dwg.save()


def write_segments_to_gcode(gcode_file_path, segment, segment_color, feedrate):
    '''Writes line segments to a gcode file with feedrate and pencil force based on segment color.'''

    with open(str(gcode_file_path) + "_spiral.gcode", 'w') as f:

        G1_flag = False
        for i, seg in enumerate(segment):
            
            if i == 0:
                f.write("G90\n")  # set to absolute positioning
                f.write(f"G0 X{seg[0][0]:.3f} Y{seg[0][1]:.3f} F{feedrate:.2f}\n")
                f.write("M4 S0\n")  # turn on pwm at 0  
            else:
                # G0 move if beginning of segment is not the same as end of previous segment
                if not np.array_equal(segment[i-1][1], segment[i][0]):
                    f.write(f"G0 X{seg[1][0]:.3f} Y{seg[1][1]:.3f} S0\n")
                    G1_flag = False
                else:
                    # convert color to grayscale value between 0 and 255
                    color_value = segment_color[i]
                    # interpolate grayscale value to pencil force
                    pencil_force = np.interp(color_value, [0, 255], [1000, 0])

                    if (G1_flag):
                        f.write(f"X{seg[1][0]:.3f} Y{seg[1][1]:.3f} S{pencil_force:.0f}\n")
                    else:
                        f.write(f"G1 X{seg[1][0]:.3f} Y{seg[1][1]:.3f} S{pencil_force:.0f}\n")
                        G1_flag = True

        f.write("M5\n")  # turn off pwm at end of job
    f.close()


def map_bitmap_to_spiral(image_filename, drawing_width_mm, 
                         spiral_center_pxl, spiral_pitch_mm, segment_length_mm):
    '''Main function to map bitmap to spiral. Generates spiral segments, maps them to pixel colors, and writes to svg and gcode.'''
    # Load image
    filename = Path(image_filename).stem
    ext = Path(image_filename).suffix
    dir = Path(__file__).parent / "data"
    file_path = dir / filename
    img = Image.open(file_path.with_suffix(ext))

    img_gray = np.array(img.convert("L")) # convert image to grayscale
    image_pixel_width, image_pixel_height = img.size

    # Calculate drawing dimensions in mm based on desired width
    mm_per_pixel = drawing_width_mm/image_pixel_width
    drawing_height_mm = image_pixel_height*mm_per_pixel
    drawing_hypotenuse_mm = np.sqrt(drawing_width_mm**2 + drawing_height_mm**2)
    
    # calculate b based on spiral pitch
    b = spiral_pitch_mm/(2*np.pi)

    # using the drawing hypotenuse to determine the number of rotations,
    # a bit hacky for all spiral center positions, but it ensures the spiral extends to the corners of the drawing
    num_rotations = drawing_hypotenuse_mm/(4*np.pi*b) + 1 # add 1 to ensure spiral extends beyond corners of drawing
    theta_end = num_rotations*2*np.pi

    seg_point = generate_spiral_segment_points(0, b, segment_length_mm, 0, theta_end)

    # translate center of spiral
    spiral_center = np.array([spiral_center_pxl[0]*mm_per_pixel, spiral_center_pxl[1]*mm_per_pixel])
    seg_point_trans = seg_point + spiral_center

    segment = create_segments_from_segment_points(seg_point_trans)

    segment_inbounds = remove_out_of_bounds_segments(segment, 0, drawing_width_mm, 0, drawing_height_mm)

    # create segment centers for each segment
    seg_center = np.mean(segment_inbounds, axis=(1))

    # determine pixel index for each segment center
    seg_center_pixel_index = np.floor(seg_center/mm_per_pixel).astype(int)

    # get color of each pixel index
    segment_color = img_gray[seg_center_pixel_index[:, 1], seg_center_pixel_index[:, 0]]

    # remove segments where pixel color is white (255) since that corresponds to no pencil mark
    segment_nonwhite = segment_inbounds[segment_color < 255]
    segment_color_nonwhite = segment_color[segment_color < 255]
    
    # sort segments in nearest neighbor order to minimize lifts
    sort_order, segment_sorted = nearest_neighbor_line_sort(segment_nonwhite)
    segment_color_sorted= segment_color_nonwhite[sort_order]

    # write to svg
    view_box = [0, 0, drawing_width_mm, drawing_height_mm]
    write_segments_to_svg(file_path, segment_sorted, segment_color_sorted, view_box)
    
    # invert y axis for gcode since svg and image coordinates have y increasing downwards 
    # but gcode has y increasing upwards
    segment_sorted[:, :, 1] = drawing_height_mm - segment_sorted[:, :, 1]
    
    write_segments_to_gcode(file_path, segment_sorted, segment_color_sorted, 250)
    

if __name__ == '__main__':

    map_bitmap_to_spiral(image_filename="margaret gym_gray.png",
                         drawing_width_mm=150,
                         spiral_center_pxl=[325,600], 
                         spiral_pitch_mm=0.7, 
                         segment_length_mm=0.5)

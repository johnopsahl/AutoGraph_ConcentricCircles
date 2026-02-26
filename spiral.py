import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import svgwrite
from PIL import Image

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

    return np.lib.stride_tricks.sliding_window_view(seg_points, 2, axis=0).transpose(0, 2, 1)

def remove_out_of_bounds_segments(segment, x_min, x_max, y_min, y_max):
    x_coords = segment[:, :, 0]  # shape (N, 2)
    y_coords = segment[:, :, 1]  # shape (N, 2)

    mask = (
        (x_coords >= x_min).all(axis=1) &
        (x_coords <= x_max).all(axis=1) &
        (y_coords >= y_min).all(axis=1) &
        (y_coords <= y_max).all(axis=1)
    )

    return segment[mask]

def create_segment_centers(segment):

    return np.mean(segment, axis=(1))

def write_segments_to_svg(filename: str, segment, view_box):
    
    dwg = svgwrite.Drawing(filename + '.svg')

    for seg in segment:          

        dwg.add(dwg.line(seg[0], seg[1], stroke=svgwrite.rgb(0, 0, 0), stroke_width=0.1))

    dwg.viewbox(minx=view_box[0], miny=view_box[1], 
                width=view_box[2], height=view_box[3])
    dwg.save()

def write_segments_and_points_to_svg(filename, segment, seg_center, view_box):
    
    dwg = svgwrite.Drawing(filename + '.svg')

    for seg in segment:          

        dwg.add(dwg.line(seg[0], seg[1], stroke=svgwrite.rgb(0, 0, 0), stroke_width=0.1))

    for pnt in seg_center:

        dwg.add(dwg.circle(center=(pnt[0], pnt[1]), r=0.1, fill='blue', stroke='blue', stroke_width=0.1))
        
    dwg.viewbox(minx=view_box[0], miny=view_box[1], 
                width=view_box[2], height=view_box[3])
    dwg.save()

# def convert_bitmap_to_spiral_drawing(image_filename: str, 
#                                      a: float, b: float, 
#                                      segment_length: float):
#     """Creates line segments along an Archimedes spiral."""

#     # Import image  
#     img = np.array(Image.open(image_filename))

#     # Set spiral center
#     spiral_center = np.array([img.size[0] // 2 + 0.5, img.size[1] // 2 + 0.5])

#     # Convert image to grayscale
#     img_gray = img.convert("L")
    
#     # Determine the angle at which the spiral should end 
#     #theta_end = calc_spiral_end(img_width, imag_height)

#     # Generate spiral segment end points
#     seg_point = generate_spiral_segment_points(a, b, segment_length, 0, theta_end)

#     # Translate segment ceneters by spiral center
#     seg_point_trans = seg_point - spiral_center

#     # Generate spiral segment center points
#     seg_center = generate_spiral_segment_points(a, b, segment_length, segment_length/2, theta_end)
    
#     # Translate segement centers by spiral center
#     seg_center_trans = seg_center - spiral_center 

#     #TODO: Make sure this snaps to pixel centers at 0.5 increments
#     # Determine pixel index neareset to each segment center point
#     seg_center_pixel = np.floor(seg_center).astype(int)

#     # Get color of each pixel index 
#     # seg_color = img[seg_center_pixel[:, 1], seg_center_pixel[:, 0]]

#     # Scale segement points according to drawing width
#     seg_point_scaled = seg_point_trans*(drawing_width/img.size[0])

#     # Write the segments to svg
#     write_segments_to_svg("spiral", seg_point_scaled, 0.1)

def display_spiral_segments(a, b, segment_length, theta_end):
    
    seg_point = generate_spiral_segment_points(a, b, segment_length, 0, theta_end)
    
    segment = create_segments_from_segment_points(seg_point)

    view_box = [-10, -10, 20, 20]
    write_segments_to_svg("spiral_segments", segment, view_box)

def clip_spiral_segments_to_image_boundaries(a, b, segment_length, theta_end):
    
    seg_point = generate_spiral_segment_points(a, b, segment_length, 0, theta_end)
    
    segment = create_segments_from_segment_points(seg_point)

    segment_filtered = remove_out_of_bounds_segments(segment, -5, 5, -5, 5)

    view_box = [-10, -10, 20, 20]
    write_segments_to_svg("spiral_segments_clipped", segment_filtered, view_box)

def display_spiral_segments_and_centers(a, b, segment_length, theta_end):
    
    seg_point = generate_spiral_segment_points(a, b, segment_length, 0, theta_end)

    segment = create_segments_from_segment_points(seg_point)

    segment_filtered = remove_out_of_bounds_segments(segment, -5, 5, -5, 5)

    #create segment centers from segments
    seg_center = create_segment_centers(segment_filtered)

    view_box = [-10, -10, 20, 20]
    write_segments_and_points_to_svg("spiral_segments_centers", 
                                     segment_filtered, seg_center, view_box)

if __name__ == '__main__':

    # convert_bitmap_to_spiral("margaret_gym.png", 0, 1.5, 5, theta_end)
    display_spiral_segments(0, 0.3, 0.5, 8*np.pi)
    clip_spiral_segments_to_image_boundaries(0, 0.3, 0.5, 8*np.pi)
    display_spiral_segments_and_centers(0, 0.05, 0.5, 64*np.pi)

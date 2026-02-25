import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import svgwrite
from PIL import Image

def calc_spiral_arc_length(theta, a):
    """Calculates arc length L of spiral r = a * theta from 0 to theta."""
    if theta == 0:
        return 0.0
    # Exact formula for Archimedean spiral arc length
    return (a / 2.0) * (theta * np.sqrt(1 + theta**2) + np.log(theta + np.sqrt(1 + theta**2)))

def find_theta_from_length(target_L, a, initial_guess):
    """Numerically solves for theta given a target arc length."""
    # Define the function whose root we want to find: f(theta) - target_L = 0
    func = lambda t: calc_spiral_arc_length(t, a) - target_L

    # root() expects the function to accept and return array-like values
    result = root(func, initial_guess)

    if not result.success:
        raise RuntimeError(f"Root finding failed: {result.message}")

    return result.x[0]

def generate_spiral_segment_points(a, b, segment_length, theta_start, theta_end):
    """Generates line segment points along an Archimedes spiral."""

    # Create an array of theta values from 0 to theta_end with a step size determined by the desired segment length
    theta = []
    theta_current = theta_start

    while theta_current < theta_end:
        length_initial = calc_spiral_arc_length(theta_current, a)
        length_target = length_initial + segment_length
        theta_new = find_theta_from_length(length_target, a, theta_current)
        theta.append(theta_new)
        theta_current = theta_new
    theta = np.array(theta)

    # Calculate the radius for each theta using the formula r = a + b*theta
    r = a + b * theta
    
    # Convert polar coordinates (r, theta) to Cartesian coordinates (x, y)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return np.columnstack((x,y))


def write_segments_to_svg(filename: str, points, stroke_width, view_box):
    
    dwg = svgwrite.Drawing(filename + '.svg')

    for i in range(len(points) - 1):          

        dwg.add(dwg.line(point[i], point[i + 1], stroke=svgwrite.rgb(0, 0, 0), stroke_width=0.1))

    dwg.viewbox(minx=view_box[0], miny=view_box[1], 
                width=view_box[2], height=view_box[3])
    dwg.save()


def convert_bitmap_to_spiral(image_filename, a, b, segment_length, theta_end):
    """Creates line segments along an Archimedes spiral."""

    img = np.array(Image.open(image_filename))

    # Generate spiral segment end points
    segment_points = generate_spiral_segment_points(a, b, segment_length, 0, theta_end)

    # Generate spiral segment center points
    center_points = generate_spiral_segment_points(a, b, segment_length, segment_length/2, theta_end)
    
    # Determine pixel index neareset to each segment center point
    center_point_pixel = np.floor(coords).astype(int)

    # Get color of each pixel index 
    center_point_color = img[points[:, 1], points[:, 0]]

    # Write the segments to svg

if __name__ == '__main__':

    convert_bitmap_to_spiral("Peppers.png", a, b, segment_length, theta_end)
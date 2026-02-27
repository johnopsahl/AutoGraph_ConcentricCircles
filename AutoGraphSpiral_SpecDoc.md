# AutoGraph Spiral

## Description
A script that maps a bitmap image to segments of an Archimedes spiral. Outputs a gcode file for the AutoGraph pencil plotter.

## CONOPS
1. User inputs reference image, drawing dimensions (in milimeters), and sprial parameters.
2. System outputs a AutoGraph pencil plotter compatible gcode file.

## Development Steps
1. ~~generate spiral segments and display as svg~~
2. ~~clip spiral coverage to image boundaries~~
3. ~~create segment centers~~
4. ~~create spiral as large as input image~~
5. ~~move center of spiral~~
6. ~~determine the number of turns required to achieve image coverage~~
7. ~~map segment centers to nearest pixel center~~
8. ~~compare spiral segment centers to source image pixel colors~~
9. ~~generate gcode and inspect in gcode viewer~~
10. ~~remove all segements with segement color 255~~
11. ~~nearest neighbor the strokes to reduce draw time~~
12. allow reverse spiral direction
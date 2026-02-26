# AutoGraph Spiral

## Description
A script that generates AutoGraph pencil plotter gcode in a archimedes spiral pattern. Based on a source image.

## CONOPS
1. User inputs reference image, drawing dimensions (in milimeters), and sprial parameters.
2. System outputs a AutoGraph compatible gcode file.

## Development Steps
1. ~~generate spiral segments and display as svg~~
2. ~~clip spiral coverage to image boundaries and display~~
3. create segement centers and nearest pixel center for each center and display
4. compare spiral segment centers to source image pixel colors and display
5. generate gcode and inspect in gcode viewer
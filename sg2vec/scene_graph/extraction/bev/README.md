# BEV Calibration Guide

<p align="center">
   <b>This calibration module should be used before extracting scene-graphs for image-based datasets.</b>
</p>

<p align="center">
   <img src='http://g.recordit.co/H333oySCra.gif' title='BEV Demo' width='' alt='BEV Demo' />
</p>

## User Controls

**Input Space (Choosing Horizon Line)**

All clicks should be done using *left click*.

**BEV Space (Choosing Warp Parameters)**

The keyboard is used to configure the bev projection, available inputs include:

* [e] creates a stronger warp
* [w] decreases warp strength

*Goal is to get orthographic projected lane lines.*

**Distance Relations (Lane Line Distances)**

Click on the top and bottom of a single lane line marking, then click once on the parallel lane line to the right or left of the first one.  This enables the code to get the relative lane length and width of the lanes and convert pixel distances to feet.

*In our experiments, the lane length and width are fixed to 10 and 12 ft. respectively following U.S. Highway Standards.*

**Output** 

The output is a json file that contains all the necessary parameters for computing distance relations.

## Instructions:

1. click on the horizon line of the image
2. scale projected image using {e, w} until lane lines are parallel
3. click (in exact order) in the projected image
   1. top and bottom of a single lane line
   2. a parallel lane line
   3. results will be saved

**Disclaimer: this module holds the assumption of a fixed camera / horizon line with relatively flat lane topology.**

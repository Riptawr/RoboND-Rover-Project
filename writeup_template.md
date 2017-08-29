## Project: Search and Sample Return
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
[image2]: ./calibration_images/example_grid1.jpg
[image3]: ./calibration_images/example_rock1.jpg
[thresh-1]: ./writeup-misc/writeup-tresh-1.png
[warped]: ./writeup-misc/warped.png

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

To allow adding different objects to be added to the map, i added a RGB range upper limit to the color_thresh function
 and added it to the truth check:
```python
def color_thresh(img, rgb_thresh_min=(160, 160, 160), rgb_thresh_max=(255, 255, 255)):
    in_range = (img[:,:,0] > rgb_thresh_min[0]) \
                & (img[:,:,1] > rgb_thresh_min[1]) \
                & (img[:,:,2] > rgb_thresh_min[2]) \
                & (img[:,:,0] < rgb_thresh_max[0]) \
                & (img[:,:,1] < rgb_thresh_max[1]) \
                & (img[:,:,2] < rgb_thresh_max[2])
                ...
```

The example outputs for regular terrain (the top) and terrain with a rock (bottom) look as follows, we can see the rock is identified correctly:

![warped-terrain][thresh-1]

#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 

 To get a correct perspective transformation from first person to bird-eye view,
 the example grid image (below) was used to determine the corners of the grid cell, right in front of the rover's camera,
 and the destinations of those points in the transformed image.

![original image with grid overlay][image2]

 A bottom offset was chosen to move the result
 in front of the rover accordingly, to account for camera location. All values were chosen empirically,
 by zooming in on the image and evaluating the accuracy manually -
 additional accuracy may be gained by working out the equations to get the correct results from a flat input picture,
 but seemed like over-engineering for this simple model.

![bird-eye view after transformation][warped]

After getting the warped (bird-eye) view from a front-camera input, we proceed with extracting objects via different RGB color thresholds:
```python
    treshed = color_thresh(warped)
    obstacles = color_thresh(warped, rgb_thresh_min=(0,0,0), rgb_thresh_max=(159,159,159))
    rocks = color_thresh(warped, rgb_thresh_min=(150,50,0), rgb_thresh_max=(255,200,50))
```

We proceed by mapping the objects coordinates to rover-centric, and then to world coordinates by rotating the rover
 so it's origin aligns with the world map's origin
```python
    #4) Convert thresholded image pixel values to rover-centric coords
    xpix, ypix = rover_coords(threshed)
    obstacle_x, obstacle_y = rover_coords(obstacles)
    rock_x, rock_y = rover_coords(rocks)

    # 5) Convert rover-centric pixel values to world coords
    scale = 8 # downscaling with data loss is intentional
    navigable_x_world, navigable_y_world = pix_to_world(xpix, ypix, data.xpos[data.count],
                                    data.ypos[data.count], data.yaw[data.count],
                                    data.worldmap.shape[0], scale)
    rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, data.xpos[data.count],
                                    data.ypos[data.count], data.yaw[data.count],
                                    data.worldmap.shape[0], scale)
    obstacle_x_world, obstacle_y_world = pix_to_world(obstacle_x, obstacle_y, data.xpos[data.count],
                                    data.ypos[data.count], data.yaw[data.count],
                                    data.worldmap.shape[0], scale)
```

Notably, the scaling factor of 8, instead of 5, when transforming from rover to world coordinates,
was found to result in a smaller map and helped with accuracy on navigable/obstacle terrain borders, which were prone to overlapping.
The fidelity difference was 5-10% compared to a scaling factor of 5.
Furthermore, incremental updates to the object's channel value (as suggested by the example code) lead to resetting every ~255 percepts,
which are common when the rover stands still for pickup or gets stuck - instead we opted for setting observed coordinates
 to their max value instead (see `update_map(y,x,ch)` below).

To increase fidelity of the result mapping further, roll and pitch of the input image were taken into account and inputs that deviate too far from being flat are skipped:
```python
    def update_map(y,x,ch):
        data.worldmap[y,x,ch] = 255


    if (data.roll[data.count] < 0.2 or data.roll[data.count] > 359.8) or (data.pitch[data.count] < 0.2 or data.pitch[data.count] > 359.8):
        update_map(obstacle_y_world, obstacle_x_world, 0)
        update_map(navigable_y_world, navigable_x_world, 2)
        update_map(rock_y_world, rock_x_world, 1)
```

To make this possible, pitch and roll were included as fields for the data class (since the recording comes with those values), which initially featured only yaw:
```python
class Databucket():
    def __init__(self):
        self.images = csv_img_list
        self.xpos = df["X_Position"].values
        self.ypos = df["Y_Position"].values
        self.yaw = df["Yaw"].values
        # Adding pitch and roll for more accurate mapping
        self.pitch = df["Pitch"].values
        self.roll = df["Roll"].values
        self.count = 0 # This will be a running index
        self.worldmap = np.zeros((200, 200, 3)).astype(np.float)
        self.ground_truth = ground_truth_3d # Ground truth worldmap
```


Finally, the output result was formed from a canvas of zeroes, with a shape to fit the necessary images in a mosaic.
A decision was made to keep the original ground truth map unchanged in the bottom left,
and add the live world-map with updates to the bottom right instead, for easier visual comparison of the two:
```python
    output_image[160:360, 320:320+data.worldmap.shape[1]] = np.flipud(data.worldmap)
```


### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.


#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  



![alt text][image3]



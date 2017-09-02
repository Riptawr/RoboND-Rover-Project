## Project: Search and Sample Return

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
[auto-1]: ./writeup-misc/auto-1.png

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

To allow adding different objects to be added to the map, a RGB range upper limit was added to the color_thresh function
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
A sample recording is available [here](https://youtu.be/HaZc5Ri0ULM)


### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and add an explanation of how and why these functions were modified as they were.

The `perception_step()` follows similar logic as the prototype in the notebook with one difference in the process of updating the map, which now tries to
reduce values for overlapping terrain/obstacle borders with a bias towards obstacles.
```python
    # 7) Update Rover worldmap (to be displayed on right side of screen)
    def update_map(y,x,ch):
        Rover.worldmap[y,x,ch] += 255
        if ch == 2:
            Rover.worldmap[y,x,0] -= 255
        if ch == 0:
            Rover.worldmap[y,x,2] -= 100
```

A percept-counter was introduced, which is used to periodically append the Rover's position to last known positions -
    a Python [deque](https://docs.python.org/3/library/collections.html#collections.deque),
    optimized for appending and removing to and from the head and tail of the queue in constant time.
    The percept-counter allows for efficient, time-delayed logic to be built on top of percepts,
     since we are running in an endless loop and would need to persist data otherwise.
     30 percepts took approx. 1 second on a Ryzen 7 @ 3.8Ghz

```python
    Rover.percept_count += 1
    # Note our position from time to time for loop detection
    if Rover.percept_count % 30 == 0:
        Rover.last_known_positions.append(Rover.pos)

    if Rover.percept_count % 30 == 0:
        first = Rover.last_known_positions[0]
        last = Rover.last_known_positions[-1]
        print("Distance travelled over last 30 percepts (~1s): {0}".format(np.linalg.norm(np.array(last)-np.array(first))))
```

Additional fields to the Rover class, in `drive_rover.py`, were needed to to allow making decisions based on being stuck or looping in place:
```python
        # Additional fields for stuck detection
        self.stuck_counter = 0
        self.evasion_mode = None
        self.last_known_positions = deque(maxlen=20)
        self.percept_count = 0
```

Based on additional fields, the decision logic got new modes for the rover and sets them when certain counters exceed their threshold:
- if we are travelling extended periods in `forward` mode, but our speed is very low -> probably stuck
- if the speed is high, but total distance travelled (as per euclidean / l2 norm) is not very large -> probably looping
```python
    if Rover.nav_angles is not None:
        if Rover.mode == "stuck":
            get_unstuck(Rover)

        if Rover.mode == "looping":
            get_unlooped(Rover)

        if Rover.mode == 'forward':
            if Rover.vel < 0.5:
                Rover.stuck_counter += 1
            else:
                Rover.stuck_counter = 0

            # If we are at a decent speed, but did not move very far over the last ~10 seconds
            if Rover.vel > 1.7:
                # Prevent from triggering before we have some positions
                if len(Rover.last_known_positions) > 4:
                    pos_hist = Rover.last_known_positions
                    dist_travelled = np.linalg.norm(np.array(pos_hist[0]) - np.array(pos_hist[-1]))
                    if dist_travelled < 8.0:
                        print(f"! Possible loop, travelled no more than {dist_travelled} over {30*10} percepts !")
                        Rover.mode = "looping"

        if Rover.stuck_counter > 90:
            Rover.mode = 'stuck'
            Rover.evasion_mode = 'forward'
            Rover.stuck_counter = 0
```

Different functions are used to get unstuck and un-looped (see `decision.py`), but they basically boil
 down to introducing corrections to the direction we are travelling in and resetting their counters / modes when done.

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

The simulator settings were `1280x1024 / Fantastic` during the tests and lead to a above 76% fidelity with ~85% of terrain,
 and 5-6 rocks located in under 300 seconds.
Time needed for mapping varies, based on the map and whether we get stuck or loop often and the current logic
 does not make intelligent decisions on where to go, therefore exploring 100% of the map is not guaranteed.

![Results of a common testrun in 300 seconds][auto-1]

A full video is available [here](https://youtu.be/yYDN1t0wVpI)

#### The main challenges of the current logic can be summarized as follows:

1. perception overly affected by roll/pitch and serves as an upper bound for mapping fidelity.

To avoid this issue, we would need to come up with equations for the destination of the image transform,
 which account for roll/pitch instead of using hardcoded destinations, based on an even picture

2. relying on only the front camera and colors to detect objects may fail in different lightning conditions. The rover arm's own shadow consistently appears in the mapping and is detected as an obstacle.

We would need additional sensors, e.g. infra-red cameras or a mapping via lidar etc. to overlay on top of the front camera, to make the detection more robust

3. Averaging visible terrain to decide on direction leads to more steering corrections on higher speed, which results in higher roll/pitch (increasing issue 1.)

Additional logic to modify the crop angle and brake distance, dynamically, based on speed may help with this issue.
One way would be mounting a range-finder and using it as an assistance to determine whether corrections are needed or not, and making them earlier,
since camera mount height and range are limited.

4. Lack of intelligent navigation / prone to re-visiting known areas.

While re-visiting areas may be useful to map from a different angle, especially on uneven terrain, or when searching hidden rocks,
more intelligent direction choice could be introduced via comparing mapped terrain with ground truth and getting unexplored areas.
The decision making could then be biased towards this direction on each step, when a driving decision is made.

Currently the decision step is based only on navigable terrain:
```python
# Set steering to average angle clipped to the range +/- 15
Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
```

An improvement could be made by adding a naive heuristic, `H` - calculated as the ideal direction - as follows:
```python
H: Union[small_left_correction, small_right_correction] = get_heading_towards_unexplored()

# Set steering to average angle clipped to the range +/- 15
Rover.steer = np.clip(np.mean((Rover.nav_angles + H) * 180/np.pi), -15, 15)
```
Or, alternatively, removing it from the correct side of the clip limit.
As a result the direction will be biased towards new terrain, given the mapping is accurate.


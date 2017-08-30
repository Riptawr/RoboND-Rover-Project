import numpy as np
import cv2


def color_thresh(img, rgb_thresh_min=(160, 160, 160), rgb_thresh_max=(255, 255, 255)):
    """Identify pixels above the threshold
    Threshold of RGB > 160 does a nice job of identifying ground pixels only

    :param img:
    :param rgb_thresh_min:
    :param rgb_thresh_max:
    :return: the binary image with pixels meeting the threshold range, set to 1
    """

    in_range = (img[:, :, 0] > rgb_thresh_min[0]) \
               & (img[:, :, 1] > rgb_thresh_min[1]) \
               & (img[:, :, 2] > rgb_thresh_min[2]) \
               & (img[:, :, 0] < rgb_thresh_max[0]) \
               & (img[:, :, 1] < rgb_thresh_max[1]) \
               & (img[:, :, 2] < rgb_thresh_max[2])

    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:, :, 0])
    # Index the array of zeros with the boolean array and set to 1
    color_select[in_range] = 1

    return color_select


def rover_coords(binary_img):
    """Define a function to convert from image coords to rover coords

    :param binary_img:
    :return:
    """
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2).astype(np.float)
    return x_pixel, y_pixel


def to_polar_coords(x_pixel, y_pixel):
    """Define a function to convert to radial coords in rover space

    :param x_pixel:
    :param y_pixel:
    :return:
    """
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles


def rotate_pix(xpix, ypix, yaw):
    """Define a function to map rover space pixels to world space

    :param xpix:
    :param ypix:
    :param yaw:
    :return: rotated x, rotated y
    """
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated


def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    """Apply a scaling and a translation

    :param xpix_rot:
    :param ypix_rot:
    :param xpos:
    :param ypos:
    :param scale:
    :return: translated x, translated y
    """

    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos

    return xpix_translated, ypix_translated


def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    """apply rotation and translation (and clipping)

    :param xpix:
    :param ypix:
    :param xpos:
    :param ypos:
    :param yaw:
    :param world_size:
    :param scale:
    :return:
    """
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)

    return x_pix_world, y_pix_world


def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))  # keep same size as input image

    return warped


def perception_step(Rover):
    """Perform perception steps to update Rover()

    :param Rover:
    :return:
    """
    Rover.percept_count += 1
    # Note our position from time to time for loop detection
    if Rover.percept_count % 30 == 0:
        Rover.last_known_positions.append(Rover.pos)

    if Rover.percept_count % 30 == 0:
        first = Rover.last_known_positions[0]
        last = Rover.last_known_positions[-1]
        print("Distance travelled over last 300 percepts (~10s): {0}".format(np.linalg.norm(np.array(last)-np.array(first))))

    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    image = Rover.img
    dst_size = 5
    bottom_offset = 6
    source = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])
    destination = np.float32([[image.shape[1] / 2 - dst_size, image.shape[0] - bottom_offset],
                              [image.shape[1] / 2 + dst_size, image.shape[0] - bottom_offset],
                              [image.shape[1] / 2 + dst_size, image.shape[0] - 2 * dst_size - bottom_offset],
                              [image.shape[1] / 2 - dst_size, image.shape[0] - 2 * dst_size - bottom_offset],
                              ])

    # 2) Apply perspective transform
    warped = perspect_transform(image, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    navigatable_terrain = color_thresh(warped)
    obstacles = color_thresh(warped, rgb_thresh_min=(0, 0, 0), rgb_thresh_max=(160, 160, 160))
    rocks = color_thresh(warped, rgb_thresh_min=(150, 50, 0), rgb_thresh_max=(255, 200, 50))

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:, :, 0] = obstacles*255
    Rover.vision_image[:, :, 1] = rocks*255
    Rover.vision_image[:, :, 2] = navigatable_terrain*255

    # 5) Convert map image pixel values to rover-centric coords
    xpix, ypix = rover_coords(navigatable_terrain)
    obstacle_x, obstacle_y = rover_coords(obstacles)
    rock_x, rock_y = rover_coords(rocks)

    # 6) Convert rover-centric pixel values to world coordinates
    xpos, ypos = Rover.pos
    scale = 6  # We err on the side of caution
    navigable_x_world, navigable_y_world = pix_to_world(xpix, ypix, xpos,
                                                        ypos, Rover.yaw,
                                                        Rover.worldmap.shape[0], scale)
    rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, xpos,
                                              ypos, Rover.yaw,
                                              Rover.worldmap.shape[0], scale)
    obstacle_x_world, obstacle_y_world = pix_to_world(obstacle_x, obstacle_y, xpos,
                                                      ypos, Rover.yaw,
                                                      Rover.worldmap.shape[0], scale)

    # 7) Update Rover worldmap (to be displayed on right side of screen)
    def update_map(y,x,ch):
        Rover.worldmap[y,x,ch] += 255
        if ch == 2:
            Rover.worldmap[y,x,0] -= 255
        if ch == 0:
            Rover.worldmap[y,x,2] -= 100

    # Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    # Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
    # Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    if (Rover.roll < 0.5 or Rover.roll > 359.5) or (Rover.pitch < 0.5 or Rover.pitch > 359.5):
        update_map(obstacle_y_world, obstacle_x_world, 0)
        update_map(navigable_y_world, navigable_x_world, 2)
        update_map(rock_y_world, rock_x_world, 1)

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    dists, angles = to_polar_coords(xpix, ypix)
    Rover.nav_dists = dists
    Rover.nav_angles = angles

    return Rover

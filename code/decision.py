import numpy as np


def get_unstuck(Rover):
    """Try to alterate between turning right and going forward until unstuck
    Will switch to forward mode once done
    :param Rover:
    :return:
    """
    # print(f"looks like we are stuck, trying: {Rover.evasion_mode}\nStuck counter {Rover.stuck_counter}")
    if Rover.evasion_mode == 'forward':
        Rover.throttle = 1
        # Rover.steer = 0
        Rover.stuck_counter += 1
        if Rover.stuck_counter > 50:
            Rover.evasion_mode = 'yaw'
            Rover.stuck_counter = 0
    elif Rover.evasion_mode == 'yaw':
        Rover.throttle = 0
        Rover.steer = -15
        Rover.stuck_counter += 1
        if Rover.stuck_counter > 30:
            Rover.evasion_mode = 'forward'
            Rover.stuck_counter = 0

    if Rover.vel > 0.6:
        Rover.mode = 'forward'
    return Rover


def get_unlooped(Rover):
    if Rover.steer > 0:
        Rover.steer = -15
        Rover.throttle = 0
    else:
        Rover.steer = 15
        Rover.throttle = 0
    Rover.mode = "forward"


def decision_step(Rover):
    """This is where you can build a decision tree for determining throttle, brake and steer
    commands based on the output of the perception_step() function

    :param Rover:
    :return:
    """
    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!
    # Example:
    # Check if we have vision data to make decisions with
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
                        print("! Possible loop, travelled no more than {0} over {1} percepts !".format(dist_travelled, 30*10))
                        Rover.mode = "looping"

        if Rover.stuck_counter > 90:
            Rover.mode = 'stuck'
            Rover.evasion_mode = 'forward'
            Rover.stuck_counter = 0

        # Check for Rover.mode status
        if Rover.mode == 'forward':
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:  
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else:  # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -18, 18)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -18, 18)
                    Rover.mode = 'forward'
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    return Rover


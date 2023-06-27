#!/usr/bin/env python

import numpy as np
import cv2
import glob
import logging
import os
import random
import sys
import time
from distutils.spawn import spawn

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

VEHICLE_NUMBER = 10
IMAGE_HEIGHT = 768
IMAGE_WIDTH = 1024
FOV = 110
OUTPUT_FOLDER_PATH = "out/rainy_night/"

def camera_callback(image, vehicle_id):
    capture = np.array(image.raw_data)
    capture = capture.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
    capture = capture[:, :, :3]  # taking out opacity channel
    cv2.imwrite("out/rainy_night/%d_%d.png" % (vehicle_id, round(time.time()*1000)), capture)

def create_camera_and_attach_to_random_vehicle_from_list(vehicles_list, world, blueprint_library):
    vehicle_id = random.choice(vehicles_list)
    vehicle_actor = world.get_actor(vehicle_id)
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('sensor_tick', '1.0')
    camera_bp.set_attribute('image_size_x', str(IMAGE_WIDTH))
    camera_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
    camera_bp.set_attribute('fov', str(FOV))
    camera_transform = carla.Transform(carla.Location(x=1.0, z=2.0))
    camera = world.try_spawn_actor(camera_bp, camera_transform, attach_to=vehicle_actor)
    return camera, vehicle_id

def calibrate_camera(camera):
    camera.blur_amount = 0.0
    camera.motion_blur_intensity = 0
    camera.motion_max_distortion = 0
    calibration = np.identity(3)
    calibration[0, 2] = IMAGE_WIDTH / 2.0
    calibration[1, 2] = IMAGE_HEIGHT / 2.0
    calibration[0, 0] = calibration[1, 1] = IMAGE_WIDTH / (2.0 * np.tan(FOV * np.pi / 360.0))
    camera.calibration = calibration
    return camera

def main():

    batch = []
    vehicles_list = []
    camera = None

    exist = os.path.exists(OUTPUT_FOLDER_PATH)
    if not exist:
        os.makedirs(OUTPUT_FOLDER_PATH)

    try:
        # 1. Setting up client and world

        client = carla.Client('localhost', 2000)

        world = client.load_world('Town10HD')

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
        synchronous_master = True

        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)

        # weather = carla.WeatherParameters(
        #     sun_altitude_angle=40,
        #     cloudiness=00.0,
        #     precipitation=00.0,
        #     wetness=00.0,
        #     fog_density=00.0)

        # world.set_weather(weather)

        world.set_weather(carla.WeatherParameters.HardRainSunset)

        # 2. Adding vehicles to the world and one sensor per vehicle

        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        for _ in range(VEHICLE_NUMBER):

            # Creating vehicle

            blueprint = random.choice(blueprint_library.filter('vehicle.*'))
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            transform = random.choice(spawn_points)

            batch.append(SpawnActor(blueprint, transform)
                         .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicle = response.actor_id
                vehicles_list.append(vehicle)

        world.tick()

        camera, vehicle_id = create_camera_and_attach_to_random_vehicle_from_list(vehicles_list, world, blueprint_library)
        if camera:
            camera = calibrate_camera(camera)
            print('Created camera for Vehicle %d' % vehicle_id)
            camera.listen(lambda image: camera_callback(image, vehicle_id))
        else:
            print("Error while creating camera!")
            return
        
        for _ in range(30):
            world.tick()

        # 3. Start the world

        ticks_number = 0
        while True:
            world.tick()

            ticks_number = ticks_number + 1
            if ticks_number % (5 * (1 / settings.fixed_delta_seconds)) == 0:   # 5s = 100 ticks in the simulation, because fixed_delta_seconds = 0.05
                                                                               # The camera will be on the car for 5 seconds before switching to a another one
                                                                               # We expect ~5 images by the car in that period, because sensor_tick = 1s
                                                                               # This is what my logic says, it sometimes makes a fool out of me :D
                # After some number of ticks, destroy old camera and create new camera and attach it to some random vehicle
                camera.stop()
                camera.destroy()

                camera, vehicle_id = create_camera_and_attach_to_random_vehicle_from_list(vehicles_list, world, blueprint_library)
                if camera:
                    camera = calibrate_camera(camera)
                    print('Created camera for Vehicle %d' % vehicle_id)
                    camera.listen(lambda image: camera_callback(image, vehicle_id))
                else:
                    print("Error while creating camera!")
                    return
                
                ticks_number = 0

    finally:
        print('Destroying actors')
        print('\nDestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
        print('\nDestroying camera')
        camera.stop()
        camera.destroy()

if __name__ == '__main__':

    main()

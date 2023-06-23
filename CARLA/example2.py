#!/usr/bin/env python

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

vehicle_number = 10

def main():

    batch = []
    vehicles_list = []
    cameras_list = []

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

        # 2. Adding vehicles to the world and one sensor per vehicle

        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        for i in range(vehicle_number):

            # Creating vehicle

            blueprint = random.choice(blueprint_library.filter('vehicle.*'))
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            transform = random.choice(spawn_points)

            vehicle = world.try_spawn_actor(blueprint, transform)

            if vehicle:

                print("Vehicle {}".format(vehicle.id))

                camera_bp = blueprint_library.find('sensor.camera.rgb')
                camera_bp.set_attribute('sensor_tick', '1.0')
                camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
                camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
                cameras_list.append(camera)
                print('Created camera for Vehicle {}'.format(vehicle.id))

                camera.listen(lambda image: image.save_to_disk('_out/%d/%06d.png' % (vehicle.id, image.frame)))

        # 3. Start the world

        while True:

            world.tick()

            for vehicle in vehicles_list:
                print("Vehicle {}".format(vehicle))

            # Break the loop after a certain duration
            if world.get_snapshot().frame == 1000:
                break

    finally:
        print('destroying actors')
        print('\ndestroying %d vehicles' % len(vehicles_list))
        for vehicle in vehicles_list: vehicle.destroy()
        print('\ndestroying %d cameras' % len(cameras_list))
        for camera in cameras_list:
            camera.stop()
            camera.destroy()

if __name__ == '__main__':

    main()

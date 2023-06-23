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

vehicle_number = 10

# def camera_callback(image, vehicle_id):
#     capture = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
#     cv2.imwrite("out/%d_%d.png" % (vehicle_id, round(time.time()*1000)), capture)

def main():

    batch = []
    vehicles_list = []
    cameras_list = []

    try:
        # 1. Setting up client and world

        client = carla.Client('localhost', 2000)

        world = client.load_world('Town03')

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

            batch.append(SpawnActor(blueprint, transform)
                         .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicle = response.actor_id
                vehicles_list.append(vehicle)

        # 3. Start the world

        for i in range(10):
            world.tick()

        # for i in range(100):
        while True:

            world.tick()

            #  if (i % 5) == 0:

                # 4. Creating sensor (camera) on a randomly chosed vehicle, in every tick of the world

            vehicle_id = random.choice(vehicles_list)

            vehicle_actor = world.get_actor(vehicle_id)

            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('sensor_tick', '1.0')
            camera_bp.set_attribute('image_size_x', '1024')
            camera_bp.set_attribute('image_size_y', '768')
            camera_bp.set_attribute('fov', '110')
            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            camera = world.try_spawn_actor(camera_bp, camera_transform, attach_to=vehicle_actor)
            if camera:
                cameras_list.append(camera)
                print('Created camera for Vehicle %d' % vehicle_id)

                camera.listen(lambda image: image.save_to_disk('_out/%d/%06d.png' % (vehicle_id, image.frame)))
                # camera.listen(lambda image: camera_callback(image, vehicle_id))

    finally:
        print('destroying actors')
        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
        print('\ndestroying %d cameras' % len(cameras_list))
        for camera in cameras_list:
            camera.stop()
            camera.destroy()

if __name__ == '__main__':

    main()

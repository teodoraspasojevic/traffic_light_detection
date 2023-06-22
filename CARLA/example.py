from distutils.spawn import spawn
import glob
import os
import sys

vehicle_number = 10

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time


def main():
    vehicle_list = []
    camera_list = []

    try:
        # 1. Setting up client and world

        client = carla.Client('localhost', 2000)

        world = client.load_world('Town10HD')

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)

        spectator = world.get_spectator()

        blueprint_library = world.get_blueprint_library()

        # 2. Adding vehicles to the world and one sensor to each one of them

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        for i in range(vehicle_number):

            # Creating vehicle

            bp = random.choice(blueprint_library.filter('vehicle'))
            transform = random.choice(world.get_map().get_spawn_points())
            vehicle_list.append(SpawnActor(bp, transform).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))
            vehicle = world.try_spawn_actor(bp, transform)

            # Creating sensor

            # camera_bp = blueprint_library.find('sensor.camera.rgb')
            # camera_bp.set_attribute('sensor_tick', '1.0')
            # camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            # camera = world.try_spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
            # if camera:
            #     camera_list.append(camera)
            #     print('created %s' % camera.type_id)

            #     camera.listen(lambda image: image.save_to_disk('_out/%d/%06d.png' % i % image.frame))

        time.sleep(15)

    finally:

        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])
        print('done.')


if __name__ == '__main__':

    main()

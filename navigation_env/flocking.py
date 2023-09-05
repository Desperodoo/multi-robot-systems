import numpy as np
import math
class flocking_ago():
    def __init__(self, config) -> None:
        self.radius = config.radius
        self.separate_cof = config.separate_cof
        self.cohesion_cof = config.cohesion_cof
        self.alignment_cof = config.alignment_cof
        self.max_force = config.max_force
        
    #planes, cohesion_weight, separation_weight, alignment_weight
    def flocking_algorithm(self,agents):
        forces = []
        angles = []

        for agent in agents:
            # Find nearby planes within radar range
            nearby_planes = self.find_nearby_planes(agent, agents)

            # Calculate the desired speed and angle for cohesion, separation, and alignment
            desired_speed_cohesion, angle_cohesion = self.calculate_cohesion(agent, nearby_planes, self.cohesion_cof)
            desired_speed_separation, angle_separation = self.calculate_separation(agent, nearby_planes, self.separate_cof)
            desired_speed_alignment, angle_alignment = self.calculate_alignment(agent, nearby_planes, self.alignment_cof)

            # Combine the desired speeds and angles
            combined_desired_speed = desired_speed_cohesion + desired_speed_separation + desired_speed_alignment
            combined_angle = angle_cohesion + angle_separation + angle_alignment
            forces.append(combined_desired_speed)
            angles.append(combined_angle)

        return forces, angles

    def find_nearby_agent(self,agent, agents):
        # Find planes within the radar range of the current plane
        nearby_planes = []

        for other_agent in agents:
            if other_agent != agent:
                distance = math.sqrt((agent.x - other_agent.x) ** 2 + (agent.y - other_agent.y) ** 2)
                if distance < self.radius:
                    nearby_planes.append(other_agent)

        return nearby_planes

    def calculate_cohesion(self,agent, near_agent, cohesion_weight):
        # Calculate the center of mass of nearby planes
        center_of_mass_x = 0.0
        center_of_mass_y = 0.0
        count = 0

        for other_agent in near_agent:
            center_of_mass_x += other_agent.x
            center_of_mass_y += other_agent.y
            count += 1

        if count > 0:
            center_of_mass_x /= count
            center_of_mass_y /= count

        # Calculate the desired speed to move toward the center of mass
        desired_speed = self.desire_speed

        # Calculate the angle (direction) towards the center of mass
        angle = 0.0
        if count > 0:
            angle = math.atan2(center_of_mass_y - agent.y, center_of_mass_x - agent.x)

        return desired_speed * cohesion_weight, angle

    def calculate_separation(self,agent, near_agent, separation_weight):
        # Calculate the avoidance angle (opposite to the average direction of nearby planes)
        avoidance_angle = 0.0
        count = 0

        for other_agent in near_agent:
            angle_to_other = math.atan2(other_agent.y - agent.y, other_agent.x - agent.x)
            avoidance_angle += angle_to_other
            count += 1

        if count > 0:
            avoidance_angle /= count

        # Calculate the desired speed to move away from nearby planes
        desired_speed = self.desired_speed

        # Calculate the angle (direction) to move away from nearby planes
        angle = avoidance_angle + math.pi  # Move in the opposite direction

        return desired_speed * separation_weight, angle

    def calculate_alignment(plane, nearby_planes, alignment_weight):
        # Calculate the average speed and direction of nearby planes
        avg_speed_x = 0.0
        avg_speed_y = 0.0
        count = 0

        for other_plane in nearby_planes:
            avg_speed_x += other_plane.v * math.cos(math.atan2(other_plane.y - plane.y, other_plane.x - plane.x))
            avg_speed_y += other_plane.v * math.sin(math.atan2(other_plane.y - plane.y, other_plane.x - plane.x))
            count += 1

        if count > 0:
            avg_speed_x /= count
            avg_speed_y /= count

        # Calculate the desired speed to match the average speed of nearby planes
        desired_speed = math.sqrt(avg_speed_x ** 2 + avg_speed_y ** 2)

        # Calculate the angle (direction) to match the average direction of nearby planes
        angle = math.atan2(avg_speed_y, avg_speed_x)

        return desired_speed * alignment_weight, angle

# 2026 Software Project—Advanced Line Follow Algorithm with Intelligent Pathfinding

William D’Olier

## Abstract

The RoboCup Junior Rescue Line competition requires teams to design an autonomous robot to navigate through an unknown path and rescue victims from a toxic chemical spill. This project focuses exclusively on the line follow and navigation portion of the task, designing an algorithm that can follow the line intelligently, mapping out the course in real-time and using logic-based reasoning to decide which path to follow. The methods used in this project are a significant improvement to existing approaches, which frequently rely on rudimentary tracking methods susceptible to line loss and navigation failures on complex routes. The proposed algorithm uses a four step process of identifying points of interest, linking intersections which are connected by lines, building a map of intersections and connecting lines identified over multiple frames, and navigating intelligently using said map.

## Requirements

The requirements listed below are taken directly from the 2026 Robocup Jr Rescue Line rules (<https://junior.robocup.org/wp-content/uploads/2026/01/RCJRescueLine2026-draft.pdf>).
Line
The black line, 1-2 cm wide, may be made with standard electrical insulating tape or printed onto paper or other materials. The black line forms a path on the floor.
Straight sections of the black line may have gaps with at least 5 cm of the straight line before each gap as measured from the shortest portion of the straight part of the line. The length of a gap will be no more than 20 cm.
Debris and Obstacles
Debris will have a maximum height of 3 mm. The organizers will not fix it to the floor. The debris consists of small materials such as toothpicks, small wooden dowels, etc.
Obstacles may include bricks, blocks, weights, and other large, heavy items. Obstacles will be at least 15 cm high and can be fixed to the floor.
An obstacle will not occupy more than one line or tile.
A robot is expected to navigate around obstacles. The robot may move obstacles, but obstacles may be very heavy or fixed to the floor. Obstacles will remain where they were moved to, even if that prevents the robot from proceeding.
Intersections
Intersection markers are green and 25 mm x 25 mm in dimension. They indicate the direction of the path the robot should follow.
The robot should continue straight ahead if there is no green marker at an intersection.
A dead end is when there are two green marks before an intersection (one on each side of the line); in this case, the robot should turn around.
The intersections are always perpendicular but may have 3 or 4 branches.
Intersection markers will be placed just before the intersection.
Other Relevant Rules
Robots must be controlled autonomously. Using a remote control, manual control, or passing information (by external sensors, cables, wirelessly, etc.) to the robot is not allowed.
Robots must be started manually by the team captain.
Any pre-mapped type of dead reckoning (movements preprogrammed based on known locations or placement of features in the field) is prohibited.
The field comprises modular tiles, which the organizers can use to make an endless number of courses for the robots to traverse.
The field will consist of 30 cm x 30 cm tiles with different patterns. The organizers will not reveal the final selection of tiles and their arrangement until the day of the competition. Competition tiles may be mounted on a hard-backing material of any thickness.
The floor is white. The floor may be either smooth or textured (like linoleum or carpet) and may have steps of up to 3 mm in height between tiles. Due to the nature of the tiles, there may be a step or gaps in the construction of the field.
At the entrance to the evacuation zone, there is a 25 mm × 250 mm strip of reflective silver tape on the floor.

## Robot Specifications

### Hardware

Item
Function
Justification
Raspberry Pi 5
Main processor—The central processor on the robot. Interacts with components via GPIO pins and I2C.
Has a high performance and efficient CPU which is required for real time image manipulation and running a full operating system.
Raspberry Pi AI HAT+ 2
AI processor—Performs tasks that require AI, such as object detection on a dedicated processor.
Reduces load on the CPU when performing AI tasks, preventing excessive lag.
Raspberry Pi Camera Module 3 Wide x2
Camera—One camera pointed straight down and one ahead for line follow and rescue.
A camera is required for advanced line follow rather than just using light/color sensors. This specific model was chosen for image quality and greater FOV, so it can see more of the line at once, and won’t miss important features on the course.
STM32F407VGT
Motor controller—Receives movement commands from the Pi over I2C, and uses PWM signals with encoder feedback and a PID loop to drive the motors.
The Raspberry Pi doesn’t have enough PWM channels for everything on the robot (4 motors, servos, and lights), so we decided we needed a dedicated MCU to offload some of the PWM requirements. We chose this particular MCU for its reliability and number of PWM channels.
MAX14870 x4
Motor driver—Takes the PWM signals from the motor controller and uses them to drive the motors at 12V at the desired speed and direction.
These were chosen because they are able to handle the stall current of our motors.
Pololu 100:1 Metal Gearmotor 20D 12V with encoders x4
Motor—The motors used to drive the robot.
Specifically chosen for high torque and durability. 4 are used to prevent slipping on slopes which was a major problem in previous competitions.
BNO086
IMU—Provides inertial data to the Pi over I2C
Improves odometry quality along with feedback from wheels, reducing drift over time.
AP64501SP-13 x3
Voltage regulator—3.3V, 5V, and 5.1V regulators for powering the PCB components and the Pi.
3.3V and 5V required for various ICs, and a dedicated 5.1V bus used to power the Raspberry Pi to handle peak current (5A).
XLM51772RHAR
Voltage regulator—12V regulator for powering the motors. Uses external fets for higher current capacity.
Required due to higher peak current requirements of motors (1.6A each, 6.4A total) which could damage the other regulators.
CYPD3175-24LQXQ
USB PD IC—Used to negotiate 5V 5A power with the Pi over USB C port
We wanted to avoid powering the Pi directly through the 5V pins since that bypasses overvoltage and overcurrent protection, and a PD IC is required to negotiate the power combination (5V 5A) required by the Pi.
STCS1APUR x2
LED current regulator—Used to provide a stable current to the front facing LEDs. Driven with a pwm signal from the Pi. Rated for 1.5A max.
Resistors can be inconsistent at high current so we chose to use a dedicated IC.
WS2812B x3
Status indicator LED—LEDs driven over SPI to display separate colours with only one control wire.
Able to be controlled with only one wire reduces complexity and footprint, and there are not enough PWM channels to drive multiple LEDs each with 3 color channels.
HC-SR04
Ultrasonic sensor—Provides a distance measurement using ultrasonic waves
The waves used are not impeded by reflective or transparent objects, but are susceptible to interference.
VL53L0X x3
TOF sensor—Distance measurement using infrared light rays.
Not as prone to interference as ultrasonic sensors, but has trouble with transparent and reflective objects.
SG90 x3
Servo motor—Used for controlling the claw and ramp gate during rescue.
Small size and ease of use (can be controlled with a simple PWM signal, doesn’t need an external motor driver).

### Software

Software
Function
Justification
Docker
Runs all of the code in a container which is predefined by a Dockerfile and is the same on every machine.
Docker allows for an environment to be set up which is consistent across all robots and requires minimal manual setup.
ROS2 Kilted
Allows for highly modular nodes which can be swapped out easily and interact via topics, services and actions. It has a vast ecosystem of packages for advanced robotics tasks.
Allows us to work on our own projects within individual packages, allows for new features to be added quickly and easily via nodes without having one massive script that does everything and is impossible to debug.
Python
Language used for the robot_core package.
The functions performed by robot_core aren’t really resource intensive, development speed is prioritised over pure optimisation.
C++
Language used for the line_follow package.
Since C++ is compiled it allows for far greater optimisation than python, which is critical for resource intensive tasks like live image processing.
Raspberry Pi OS

## Typical Approaches

The most basic line follow algorithm typically used for this purpose involves only two colour sensors faced down towards the front of the robot, usually less than a centimetre above the ground and only a few centimetres apart, such that the distance between them is slightly wider than the width of the line being followed. The robot is then positioned with the line between the two sensors, and is programmed to drive forward while keeping the line between the two sensors, by turning in the opposite direction when it senses black (the line) on one of the sensors. This works well enough for a simple, solid black line with constant width and no intersections, but becomes increasingly unreliable and difficult to program when additional rules are added, such as intersections where the robot must follow the path indicated by a green square. There is also the issue that if the line falls out of the narrow section between the two sensors, it is impossible to find it again, which is especially problematic on sharp corners.

Another slightly more complicated, yet still relatively simple approach is to position a camera at the front of the robot facing down towards the ground, along with a source of light, and use the images produced, filtered through a lightness threshold to differentiate the line from its surroundings, to determine the centre of mass of all black pixels in the image. To make this even more reliable the image could first be processed to find all contours, and use the centre of mass of the largest to filter out any unwanted noise. This tends to be far more accurate than the previous approach at following the line, but can struggle at intersections and especially when features surrounding the line require the robot to perform a special action, such as performing a u-turn when there is a green square on both sides of the line. Since this algorithm cannot actually isolate the individual lines themselves, intersections, or any other features, it cannot fulfill any non-trivial demands that require a deeper understanding of the layout of the track.

## The Ideal Solution

For a robot to actually navigate a track, rather than just mindlessly follow wherever the largest line happens to be at any given moment, there are some basic, high level requirements. First, it must be able to identify any points of interest in a particular frame, such as line intersections and green squares. Next, it must be able to record these points of interest and their position relative to the robot, which means the robot must know its own position, whether by precise odometry or matching points between frames, and continuously build a map of the track. Finally, the robot must determine what path it will navigate based on pre-defined rules, and knowing where its starting position is, create a path of nodes to follow, whilst continuously mapping unseen portions of the track. This makes it far more computationally expensive than other approaches. It must be able to do all this in real time, so any algorithms must be efficient and the hardware used should be powerful enough to run them.

This approach can be broken down into the following main high-level individual problems which must be solved in order to have a complete algorithm:

- Identifying points of interest
- Linking connected nodes
- Building a map of nodes
- Pathfinding using map

Each of these problems have various potential solutions which are explored in this paper, and eventually the one that works best for each will be chosen for the final robot.

## Breaking Down Each Problem

### Identifying Points of Interest

Out of all the problems identified this is perhaps the simplest, but it still presents some unique challenges. There are a few possible ways this can be done, which have implications for what approaches can be taken in the following problems. One way is to skeletonise the image, which means to shrink every line to be only one pixel wide, and can be done using the opencv python library. Once this has been done, any pixel with a value of 1 is part of the skeletonised line, and any pixel with a value of 0 is not. In the 3x3 grid surrounding a pixel of value 1, if exactly 2 also are of value 1, then the pixel is part of a line segment. If there is only one surrounding pixel with that value, then it must be the end of a line segment. If there are more than two, then it must be an intersection between two or more lines. The basic logic is to iterate over every pixel in the frame, and apply this logic to determine where the intersections and ends of lines lie on the image. As well as line features, the algorithm must also be able to identify other points of interest such as green squares or silver. Detecting colour can be done by applying a threshold to the original image, finding all contours, and their centres of mass.

### Linking Connected Nodes

The method used for finding how nodes are connected depends largely on the desired method of storing this information. For the purposes of this project, there are two arrays, one of nodes and one of connecting edges. This was chosen over other methods, as it provides much greater data integrity and the performance tradeoffs are minimal. After nodes are found, the script traces the skeletonized image from node to node, recording each unique edge. After this, the script removes small lines and merges close intersections since sometimes intersections are recorded as 2 very close nodes. Finally, the script removes all nodes that are no longer a part of the final graph.

### Building a Map of Nodes

This stage requires nodes to be identified and linked between frames, and must be able to handle nodes disappearing off frame and phantom nodes appearing from random noise. Connecting nodes between frames is done using a kalman filter to predict where the tracked nodes should be based on odometry data, and then using the hungarian algorithm to match nodes in a frame with known nodes. To eliminate phantom nodes, they are not trusted immediately but are assigned an age attribute, and are only added to the map after it reaches a certain value. If the kalman filter predicts a node will leave the frame, it is removed from the graph.

### Pathfinding Using Map

When the script begins, it locates the edge closest to the centre and follows towards the node higher up in the frame. When calculating which path to follow, the algorithm follows along an edge until it reaches a node. Once at a node, if it is an intersection, the algorithm locates all green contours and finds their angle and distance relative to the node. If they are close enough, their direction is used to determine which edge to follow next based on the competition rules. If no green square is detected, it finds the edge closest to 180º and follows that one. If the node is an end node, the algorithm creates a line pointing out from the edge and finds the closest node to follow. If there are no close nodes, it continues straight until one is found.

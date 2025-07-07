# Isotropic-6-degree-of-freedom-bionic-arm
Initial analysis of 6 DOF robotic arm in MATLAB, CAD modeling in SolidWorks 2015, and control implementations in C++ using ROS.

Analysis:
- Contains MATLAB functions for forward kinematics of robot.
- mainAnalysis.m contains torque analysis.

CAD:
- Contains CAD models designed in SolidWorks 2015.
  - Designed to be manufactured by cutting from laser cutter and adhered using weld-on acrylic cement (link below).
  - https://www.amazon.com/Weld-Acrylic-Adhesive-Applicator-Bottle/dp/B0096TWKCW.
  - Fasteners are intentionally omitted.

Controls [Not yet]:
- Contains a catkin workspace that includes:
  - (initial) Basic servo controls using .ino files.
  - (planned) ROS packages for motion/trajectory planning, visualization in RViz, and feedback control.
 
```
robotic-arm-vision/
├── src/                    # Core application code
│   ├── vision/             # Computer vision modules
│   ├── control/            # Robotic arm control
│   ├── communication/      # Hardware communication
│   └── utils/              # Utility functions
├── hardware/               # Physical components
├── arduino/                # Microcontroller firmware
├── examples/               # Demo applications
├── tests/                  # Unit testing framework
├── docs/                   # Comprehensive documentation
├── scripts/                # Setup and utility scripts
└── config/                 # Configuration files
```

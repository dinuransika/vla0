# SO-101 Robot Experiment

## Setting up the robots

- It's a good idea to use tape to mark the final robot's position.

<img src="../../assets/follower.jpg" style="width: 50%">

- Follower power supply.

<img src="../../assets/Follower Power Supply.jpg" style="width: 50%">

- Leader power supply.

<img src="../../assets/Leader Power Supply.jpg" style="width: 50%">

## Environment Setup

- Set up the environment following the [Lerobot Documentation](https://huggingface.co/docs/lerobot/en/so101).
- Follow the documentation above to set motor IDs and baudrates for both the leader and the follower arm.
- If the motors are not connecting, make sure power is available to the robot from the power supply and check the wire.
- For calibration, refer to the video in the official documentation.
- After calibrating both the leader and the follower, keep a backup of these calibration values in case you initialize the calibration sequence by accident and lose the previous calibration.
- JSON files should be in this folder: `~/.cache/huggingface/lerobot/calibration`
- If you want to replicate the results on a different machine, just copy the files in the folder structure. When prompted by the lerobot module, choose to use the existing calibration data.
- Once you find the port (e.g. `/dev/tty.usbmodem5A680130541`) for a particular robot, it doesn't change even if you use a different physical port.

### TODO

- The original robot callibration code assumes the wrist roll motor, the one rotating the gripper is a free to rotate infinitely, even though the robot has physical constraints. It would be beneficial to add callibration logic to the motor as well.

## Teleoperating the robot 

```bash
lerobot-teleoperate    --robot.type=so101_follower    --robot.port=/dev/tty.usbmodem5A680130541    --robot.id=my_awesome_follower_arm    --teleop.type=so101_leader    --teleop.port=/dev/tty.usbmodem5A7A0576571  --teleop.id=my_awesome_leader_arm
```

- Be mindful to make the leaders position closely match to the followers posotion when starting any teleopration command as seeing below.

<p float="left">
  <img src="../../assets/follower%20initial%20position.jpg" width="49%" />
  <img src="../../assets/leader%20intial%20position.jpg" width="49%" />
</p>

>⚠️ **Warning:** After callibration when you first trying to teleoperate the robot it is advicable to be ready to kill the power to the follower just in case it tries to move towards a physical constraint. 

## Cameras

```bash
(lerobot) dinura.dissanayake@LM006237 ~ % lerobot-find-cameras
--- Detected Cameras ---
Camera #0:
  Name: OpenCV Camera @ 0
  Type: OpenCV
  Id: 0
  Backend api: AVFOUNDATION
  Default stream profile:
    Format: 16.0
    Fourcc: 
    Width: 1920
    Height: 1080
    Fps: 15.0
--------------------

Finalizing image saving...
Image capture finished. Images saved to outputs/captured_images
```
You will be able to find all the connected cameras using this command and get their respective IDs. 

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5A680130541 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{
    front: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 30},
    wrist: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}
  }" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5A7A0576571 \
  --teleop.id=my_awesome_leader_arm \
  --display_data=true \
  --dataset.repo_id=dinura/so101_pickup_lego_square \
  --dataset.root=/path_to_dataset \
  --dataset.num_episodes=50 \
  --dataset.single_task="Pick up the square Lego block and place it in the container" \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \ 
  --resume=true # remove this if you are collecting data from scratch
  ```

  The above command will open the rerun window where you will be able to see the two camera POVs and other details. 

  <img src="../../assets/setup_back.jpg" style="width: 100%">
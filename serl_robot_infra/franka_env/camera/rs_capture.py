import numpy as np
import depthai as dai


class RSCapture:
    def get_device_serial_numbers(self):
        devices = dai.Device.getAllAvailableDevices()
        return [d.getMxId() for d in devices]

    def __init__(self, name, serial_number, dim=(640, 480), fps=15, depth=False, exposure=40000):
        self.name = name
        available_devices = self.get_device_serial_numbers()
        assert serial_number in available_devices, f"Device {serial_number} not found. Available: {available_devices}"
        self.serial_number = serial_number
        self.depth_enabled = depth
        self.dim = dim
        self.fps = fps

        # Create pipeline
        self.pipeline = dai.Pipeline()

        # Create ColorCamera node
        cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(fps)

        # Set manual exposure (exposure time in microseconds)
        cam_rgb.initialControl.setManualExposure(exposure, 800)  # exposure_time_us, iso_sensitivity

        # Create output for RGB
        xout_rgb = self.pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")

        # Determine if we need to scale the RGB output
        if dim[0] == 1920 and dim[1] == 1080:
            cam_rgb.video.link(xout_rgb.input)
        else:
            # Create ImageManip node to resize
            manip_rgb = self.pipeline.create(dai.node.ImageManip)
            manip_rgb.initialConfig.setResize(dim[0], dim[1])
            manip_rgb.setMaxOutputFrameSize(dim[0] * dim[1] * 3)
            cam_rgb.video.link(manip_rgb.inputImage)
            manip_rgb.out.link(xout_rgb.input)

        if self.depth_enabled:
            # Create stereo depth node
            stereo = self.pipeline.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Align depth to RGB
            stereo.setOutputSize(dim[0], dim[1])

            # Create left and right mono cameras
            mono_left = self.pipeline.create(dai.node.MonoCamera)
            mono_right = self.pipeline.create(dai.node.MonoCamera)
            mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
            mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
            mono_left.setFps(fps)
            mono_right.setFps(fps)

            # Link mono cameras to stereo node
            mono_left.out.link(stereo.left)
            mono_right.out.link(stereo.right)

            # Create output for depth
            xout_depth = self.pipeline.create(dai.node.XLinkOut)
            xout_depth.setStreamName("depth")
            stereo.depth.link(xout_depth.input)

        # Connect to device
        device_info = dai.DeviceInfo(self.serial_number)
        self.device = dai.Device(self.pipeline, device_info)

        # Create output queues
        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        if self.depth_enabled:
            self.q_depth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    def read(self):
        try:
            # Get RGB frame
            in_rgb = self.q_rgb.get()
            if in_rgb is None:
                return False, None

            image = in_rgb.getCvFrame()

            if self.depth_enabled:
                # Get depth frame
                in_depth = self.q_depth.get()
                if in_depth is None:
                    return False, None

                depth = in_depth.getFrame()
                # Expand depth dimension to match expected format
                depth = np.expand_dims(depth, axis=2)
                # Concatenate RGB and depth
                return True, np.concatenate((image, depth), axis=-1)
            else:
                return True, image

        except Exception as e:
            print(f"Error reading frame: {e}")
            return False, None

    def close(self):
        self.device.close()

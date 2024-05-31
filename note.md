sudo xhost +  &&sudo docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name foundationpose  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v/home/siyuan/test/FoundationPose:/foundation -v /mnt:/mnt -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp  --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE zsy_founda_d435

# docker环境指南

## 说明 
本docker基于https://github.com/NVlabs/FoundationPose/issues/27构建，能够已cuda12.1运行foundationpose，
### 环境

* 基于cuda12.1创建
* 原系统为cuda12.1+ubuntu22.04

### docker内部

* 已换源至中国科学技术大学源

* 环境基于conda构建，my即为完成foundationpose配置的环境 

* realsense SDK自行编译完成安装，位于librealsense文件夹下

* 代码请自行下载
## 使用指南
1. 安装cuda12.1驱动 

https://developer.nvidia.com/cuda-12-1-0-download-archive

2. 查看realsense相机端口 
sudo apt-get install v4l-utils

v4l2-ctl --list-devices

eg：

base) siyuan@4090:~$ v4l2-ctl --list-devices
Intel(R) RealSense(TM) Depth Ca (usb-0000:00:14.0-7):
	/dev/video0
	/dev/video1
	/dev/video2
	/dev/video3
	/dev/video4
	/dev/video5
	/dev/media0
	/dev/media1

3. 删除可能存在的docker ：

    sudo docker rm -f foundationpose

3. 启动docker

eg：

    sudo xhost + && sudo docker run --gpus all \
    --device /dev/video0:/dev/video0 \
    --device /dev/video1:/dev/video1 \
    --device /dev/video2:/dev/video2 \
    --device /dev/video3:/dev/video3 \
    --device /dev/video4:/dev/video4 \
    --device /dev/video5:/dev/video5 \
    --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name foundationpose \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v /home/siyuan/code/fouda_test/FoundationPose:/foundation \
    -v /mnt:/mnt \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp:/tmp \
    --ipc=host \
    -e DISPLAY=$DISPLAY \
    -e GIT_INDEX_FILE \
    zsy_founda_d435



        说明：

        **/home/siyuan/test/FoundationPose** 为电脑上项目所在地址

        sudo xhost +: 这部分是用来禁用 X11 服务器的访问控制，允许所有客户端连接。
        sudo docker run: 开始定义 Docker 运行命令。
        --gpus all: 允许 Docker 容器使用所有可用的 GPU。
        --device /dev/videoN:/dev/videoN: 映射多个视频设备到容器。
        --env NVIDIA_DISABLE_REQUIRE=1: 设置环境变量，可能是用来覆盖某些 NVIDIA 的默认要求。
        -it: 保持 STDIN 打开并分配一个终端。
        --network=host: 使用宿主机的网络。
        --name foundationpose: 指定容器名称。
        --cap-add=SYS_PTRACE, --security-opt seccomp=unconfined: 提高容器的权限，对调试有帮助。
        -v /path/on/host:/path/in/container: 挂载宿主机的目录到容器内。
        --ipc=host: 使用宿主的 IPC 命名空间。
        -e DISPLAY=$DISPLAY, -e GIT_INDEX_FILE: 设置环境变量，尤其是 DISPLAY 变量对于 GUI 程序是必需的。
    
4. 在终端使用realsense-viwer验证能否使用相机
5. 使用nvidia-smi指令和nvcc -V验证能否使用显卡
6. 第一次使用需要编译12.1版本的pytorch

    bash ./build_all.sh

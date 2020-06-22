# ywj_drl
#毕业论文环境和算法部分代码  
**1.配置环境依赖**  
``` 
conda env create -f environment.yaml  
``` 
**2.运行**  
_要先启动仿真环境才能运行下面语句，仿真环境和drl运行在不同环境_  
``` 
conda activate tf2  #进入虚拟环境
python run_td3.py   #进入对应目录，运行脚本
``` 

**3.注意**  
如果想重新训练，修改examples/run_td3.py中变量***Load***为False，examples/gazebo_env/environment_stage_3.py的***self.test***为False

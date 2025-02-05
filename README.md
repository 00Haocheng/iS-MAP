# iS-MAP： Neural Implicit Mapping and Positioning for Structural Environments（ACCV 2024）[[Paper]](https://link.springer.com/chapter/10.1007/978-981-96-0969-7_22)
![image](pipline/pipline.jpg)
Overview of the system: We sample the 3D points along the ray from each pixel and then encode the sample points by hybrid hash and multi-scale feature plane, and decode them to the TSDF value $s_i$ and the TSDF feature $h_i$ by the geometric decoder  $\varphi_{g}$. Considering the consistency of geometry and appearance, $h_i$ is then concatenated with the feature plane encoding $\rho(x_{i})$ to predict the raw color $c_i$ by appearance decoder  $\varphi_{c}$.  After TSDF volume rendering, the scene representation is optimized by minimizing  sdf loss $L_{sdf}$, smooth loss $L_{smo}$, depth loss $L_d$ , color loss $L_c$ and structural consistency loss $L_{con}$ in the mapping thread. Additionally, we also added Manhattan matching loss $L_{coor}$ to the tracking thread to further optimize the camera pose.
## Installation ##
You can create an anaconda environment called ismap. Please install libopenexr-dev before creating the environment.
``` 
sudo apt-get install libopenexr-dev
conda env create -f environment.yaml
conda activate ismap
``` 
You will then need to install tiny-cuda-nn to use the hash grid. We recommend installing it from [source code.](https://github.com/nvlabs/tiny-cuda-nn)
``` 
cd tiny-cuda-nn/bindings/torch
python setup.py install
``` 
## Download Dataset & Data preprocessing ##
Download the data as below and the data is saved into the ./Datasets/Replica folder.
``` 
bash scripts/download_replica.sh
``` 
For running iS-MAP, you should generate line mask with LSD first. We provide a simple preprocessing code to perform LSD segmentation. For example, for room0 scene, after downloading the Replica dataset, you can run 
``` 
python preprocess_line.py 
``` 
The line mask images will be generated in the root path of dataset and the folder named ```./line_seg```. 

Alternatively, we recommend directly downloading the preprocessed dataset including the ```./line_seg``` folder [here.](https://drive.google.com/drive/folders/1vU4aisJtXd2-svHKarMhPx5AMVQHfxgg?usp=sharing)

## Run ##
After downloading the data to the ./Datasets folder, you can run iS-MAP:
``` 
python -W ignore run.py configs/Replica/room0.yaml
``` 
The rendering image and reconstruction mesh are saved in ```$OUTPUT_FOLDER/mapping_vis ``` and``` $OUTPUT_FOLDER/mesh```. The ``` final_mesh_eval_rec_culled.ply```  means mesh culling the unseen and occluded regions.

## Evaluation ##
### Average Trajectory Error ###
To evaluate the average trajectory error. Run the command below with the corresponding config file:
``` 
python src/tools/eval_ate.py configs/Replica/room0.yaml
``` 
### Reconstruction Error ###
We follow the evaluation method of [Co-SLAM](https://github.com/HengyiWang/Co-SLAM), remove the unseen areas and add some virtual camera positions to balance accuracy and prediction ability. For detailed evaluation code, please refer to [here.](https://github.com/JingwenWang95/neural_slam_eval)
## Acknowledgement ##
Thanks to previous open-sourced repo: [ESLAM](https://github.com/idiap/ESLAM), [Co-SLAM](https://github.com/HengyiWang/Co-SLAM), [NICE-SLAM](https://github.com/cvg/nice-slam)
## Citing ##
If you find our work useful, please consider citing:
``` 
@inproceedings{wang2024map,
  title={iS-MAP: Neural Implicit Mapping and Positioning for Structural Environments},
  author={Wang, Haocheng and Cao, Yanlong and Shou, Yejun and Shen, Lingfeng and Wei, Xiaoyao and Xu, Zhijie and Ren, Kai},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  pages={747--763},
  year={2024}
}
``` 
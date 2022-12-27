## Linear Regression vs. Deep Learning: A Simple Yet Effective Baseline for Human Body Measurement

**NOTE: The due date for the next set of improvements is June 30th (see Updates/Work-In-Progress section).**

We show that the simple linear regression model performs comparably to the state-of-the-art for the task of human body measurement. The input to the model only consists of the information that a person can self-estimate, such as height and weight. The simplicity of the proposed regression model makes it perfectly suitable as a baseline in addition to the convenience for applications such as the virtual try-on. To improve the repeatability of the results of our baseline and the competing methods, we provide guidelines toward standardized body measurement estimation. An overview of our approach can be summarized in an image:

![https://github.com/kristijanbartol/linear-3d-humans/blob/master/assets/overview.png](https://github.com/kristijanbartol/linear-3d-humans/blob/main/assets/overview.png)


## Citation

The code is a supplementary for our journal paper. Please cite it in your research:

```
@article{Bartol:Linear-3D-Humans, 
    title={Linear Regression vs. Deep Learning: A Simple Yet Effective Baseline for Human Body Measurement}, 
    volume={22}, 
    ISSN={1424-8220}, 
    url={http://dx.doi.org/10.3390/s22051885}, 
    DOI={10.3390/s22051885}, 
    number={5}, 
    journal={Sensors}, 
    publisher={MDPI AG}, 
    author={Bartol, Kristijan and Bojanić, David and Petković, Tomislav and Peharec, Stanislav and Pribanić, Tomislav}, 
    year={2022}, 
    month={Feb}, 
    pages={1885} 
}
```

## Installation

To install the required packages, please create new virtual environment and use requirements.txt (for simplicitly, I dumped all currently installed packages to requirements.txt). 
In the future, will provide more accurate installation instruction.

## How to run

To run the demo:

```
python3 demo.py
```

The expected input/output:

```
Enter your gender ([male, female]):male
Enter your height [in meters] and weight [in kg]: 1.72 65
head_circumference: 55.32cm
neck_circumference: 37.31cm
shoulder_to_crotch: 64.38cm
chest_circumference: 89.44cm
waist_circumference: 84.36cm
hip_circumference: 92.78cm
wrist_circumference: 16.26cm
bicep_circumference: 28.05cm
forearm_circumference: 26.33cm
arm_length: 50.94cm
inside_leg_length: 73.44cm
thigh_circumference: 56.03cm
calf_circumference: 34.45cm
ankle_circumference: 21.23cm
shoulder_breadth: 35.31cm
```

![screenshot](https://github.com/kristijanbartol/linear-3d-humans/blob/main/assets/demo-screenshot.png)

## Updates / Work-In-Progress

- [X] Prepare rudimentary demo

- [ ] Clean the code (v0.1)

- [ ] Prepare-the-training-data instructions

- [ ] Finish the documentation

- [ ] Provide inference scripts and instructions

- [ ] Create a simple Python UI (v0.2)

- [ ] Add source code of the Kivy mobile that uses the fitted coefficients

## License

Software Copyright License for **non-commercial scientific research purposes**.
Please read carefully the [terms and conditions](https://github.com/vchoutas/smplx/blob/master/LICENSE) and any accompanying documentation before you download and/or use the SMPL-X/SMPLify-X model, data and software, (the "Model & Software"), including 3D meshes, blend weights, blend shapes, textures, software, scripts, and animations. By downloading and/or using the Model & Software (including downloading, cloning, installing, and any other use of this github repository), you acknowledge that you have read these terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Model & Software. Any infringement of the terms of this agreement will automatically terminate your rights under this [License](./LICENSE).


## Acknowledgments


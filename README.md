# Analyzing the Consequences of High-Intensity Light and Strategies for Defending Against Adversarial Attacks

## Description

This repository documents a research project that investigates the effects of high-intensity light on the performance of object detection models, specifically, the YOLO5 model. The project also explores the concept of adversarial attacks with light and looks at possible defense mechanisms. The ultimate aim of this research is to understand and improve the model's performance under varying lighting conditions, with a particular focus on low-light and nighttime scenarios.

## Methodology

1. **Data Preparation**: We started by curating a custom dataset, composed of images captured under a wide range of lighting conditions. 

2. **Model Training**: This custom dataset was then used to train the YOLO5 model.

3. **Performance Evaluation**: The model's performance was evaluated under different light conditions. Initial results indicated suboptimal performance under low-light and nighttime conditions.

4. **Model Refinement**: To address the performance issue, we enriched the training dataset with additional low-light and nighttime images.

5. **Experiment with Light Noise**: We introduced light noise into the model at calculated and random positions. The calculated positions were determined from the gradient between the input and the loss. 

6. **Adversarial Attack Test**: To validate our light noise introduction, we applied a conventional adversarial attack on the model. 

7. **Journal Writing**: We are in the process of documenting our research findings in a journal article, where I am listed as the first author. 

## Findings

The introduction of light noise had a minimal impact on the performance of the model. This finding was confirmed through a standard adversarial attack. The paper, soon to be published, will provide a comprehensive overview of our research and its outcomes.

## Contributions

Contributions, issues, and feature requests are welcome. Feel free to check the [issues page](LINK_TO_ISSUES) to see how you can contribute to this project.

## License

This project is licensed under the terms of the MIT license.

## Contact

Feel free to reach out if you have any questions, or ideas, or need help with the project.

This repo was built upon the code from this [repo](https://github.com/ultralytics/yolov5)

* GitHub: [@shah-imran](https://github.com/Shah-imran)
* Email: mdshahimranshovon@gmail.com

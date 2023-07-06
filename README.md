Effects of High-Intensity Light, Adversarial Attack with Light and Its Defence
Description
This repository documents a research project that investigates the effects of high-intensity light on the performance of object detection models, specifically, the YOLO5 model. The project also explores the concept of adversarial attacks with light and looks at possible defense mechanisms. The ultimate aim of this research is to understand and improve the model's performance under varying lighting conditions, with a particular focus on low-light and nighttime scenarios.

Methodology
Data Preparation: We started by curating a custom dataset, composed of images captured under a wide range of lighting conditions.

Model Training: This custom dataset was then used to train the YOLO5 model.

Performance Evaluation: The model's performance was evaluated under different light conditions. Initial results indicated suboptimal performance under low-light and nighttime conditions.

Model Refinement: To address the performance issue, we enriched the training dataset with additional low-light and nighttime images.

Experiment with Light Noise: We introduced light noise into the model at calculated and random positions. The calculated positions were determined from the gradient between the input and the loss.

Adversarial Attack Test: To validate our light noise introduction, we applied a conventional adversarial attack on the model.

Journal Writing: We are in the process of documenting our research findings in a journal article, where I am listed as the first author.

Findings
The introduction of light noise had a minimal impact on the performance of the model. This finding was confirmed through a standard adversarial attack. The paper, soon to be published, will provide a comprehensive overview of our research and its outcomes.

Contributions
Contributions, issues, and feature requests are welcome. Feel free to check the issues page to see how you can contribute to this project.

License
This project is licensed under the terms of the MIT license.

Contact
Feel free to reach out if you have any questions, ideas, or need help with the project.

GitHub: @shah-imran
Email: email@example.com

CURA AI
Description
CURA AI is an advanced tool designed to analyze MRI and CT scans of brain tumors. It assigns detailed annotations to each scan, helping medical professionals by providing clearer and more actionable insights from medical imaging. This tool is part of a broader initiative aimed at improving the efficiency and accuracy of medical imaging analysis.
Installation Instructions
	1	Clone the Repository:  git clone https://github.com/myselfadityaranjan/curaAIproject.git
	2	Navigate to the project directory  
	3	Create and Activate a Virtual Environment:  python3 -m venv venv >	source venv/bin/activate
	4	Install Dependencies:  pip3 install -r requirements.txt  
	5	Download Additional Resources: If there are any additional resources or datasets needed, download them as instructed in the project documentation.
Usage Instructions
	1	Run the Application:
	◦	Start the application by running:  python app.py  
	2	Open the Web Interface:
	◦	Open a web browser and navigate to the local HTML page provided in the project. 
	3	Input Files:
	◦	Use the web interface to upload MRI or CT scan files. The tool will analyze the scans and provide annotations based on the input.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For any questions or assistance, please reach out to:
	•	Aditya Ranjan: myselfadityaranjan@gmail.com
Acknowledgements
	•	TCIA - The Cancer Imaging Archive: For providing the datasets used in this project:
	◦	Schmainda KM, Prah M (2018). Data from Brain-Tumor-Progression. The Cancer Imaging Archive. DOI: 10.7937/K9/TCIA.2018.15quzvnb
	◦	Barboriak, D. (2015). Data From RIDER NEURO MRI. The Cancer Imaging Archive. DOI: 10.7937/K9/TCIA.2015.VOSN3HN1

 IF YOU WISH TO ACCESS THE FILES "images.npy" and "labels.npy":
1. Download the mentioned datasets from TCIA
2. Place that in the directory of this project post installation
3. Run the "main.py" script in "src"
4. The files "images.npy" and "labels.npy" should appear in the main directory

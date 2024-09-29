Customer Segmentation using Self-Organizing Maps (SOM)
This project uses a Self-Organizing Map (SOM) to cluster customers based on their purchasing data and behavior. The dataset includes information about customer purchases, product 
categories, time spent on the site, and other related features.

Requirements:
 Python 3.x
Required libraries:
 pandas
scikit-learn
 minisom
matplotlib
You can install the required libraries by running:

bash
pip install pandas scikit-learn minisom matplotlib
Steps to Run the Code:
Clone or download the repository and ensure that the dataset (purchase_data_exe.csv) is in the project folder.

Run the Python script:

bash
python customer_segmentation_som.py
The script will train the SOM using the data, and a visualization of the customer segments will be displayed as a heatmap.

Dataset:
The dataset used contains features such as:

Product category
Purchase value
Time on site
Number of clicks
These features are used to group customers into segments based on similar purchasing patterns.

Output:
A distance map is generated showing customer clusters. The map helps visualize how customers are grouped based on similarities.

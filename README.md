# cmpt310-rcpt-scan

CMPT 310 - D200 Introduction to Artificial Intelligence, Summer 2025

Group 6

Daniel Shi, Farzan Ustad and Andy Wang


**Overview**

The project finds the total from a receipt photo and extracting that into text, this can also reads a folder of receipts and export a csv file. This is useful when users may want to keep track of receipt totals but do not want to manually input it every time.


## Structure of the Project

![System Diagram](/System%20DiagramV2.png)

** Hyper Parameter Trainin**
We have our cross validation training and preprocessing functions with hyper parameters on a seperate branch called IncreaseAcc. 

**Image Data Set**

All the receipt images we used in our project were downloaded from Kaggle. Each image has a JSON file containing important information such as "total" and "dates." We used the pre-trained solution to validate and improve our algorithm and find the accuracy.

**Pre-Processing**

Preprocessing was quite necessary because raw receipt images often contain a lot of noise, skewness, uneven lighting, and low contrast, which all together can cause our model to misread characters and return a lower accuracy. 

By applying gray scale, binarization, and skew correction, we were able to enhance text clarity and improve the contrast between characters and the background of the receipt.


**Text Detection**

The pytesseract.image_to_data function detects the text and its relative location and to create a table. 

The program then sort the table and group columns that have the same block number and line number. By reading each line and compare with two list of keywords to include or exclude array with the possible correct total in it.


**Out Put**

After run the fullScaleTest.py file, it will generate a csv files with extracted total amount and the correct total amount. 




## Installation Guide
Download the repository:


```bash
git clone https://github.com/andyrzwang/cmpt310-rcpt-scan
```

**Python Version**

This project is build on Python version 3.10.0. 


**Required Python Libraries**

- opencv-python
- pytesseract
- Pillow
- datetime
- scipy
- pandas
- progressbar2
- json

**Deployment and usage Guide**

The full scale test function is in the textExtraction folder.



## Acknowledgements

Raw data is downloaded from Kaggle.com

- [Receipt Dataset for information extraction](https://www.kaggle.com/datasets/dhiaznaidi/receiptdatasetssd300v2/data)

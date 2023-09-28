# Sudoku Solver with OCR using Python and OpenCV

<br/>

## Description
This project demonstrates solving Sudoku puzzles using Python and OpenCV. Sudoku is a popular logic-based number placement game, and this repository provides a Python implementation of a Sudoku solver. 

<br/>

## Model for Digit Recognition
OCR stands for Optical Character Recognition. It is a technology that is used to recognize and convert printed or handwritten text, as well as characters and symbols, into machine-readable text. OCR software and systems are designed to process images or scanned documents that contain text, extracting the textual content for various purposes.

The `model-OCR.h5` file in this repository contains a trained machine learning model for digit recognition. This model is used by the Sudoku solver to recognize and solve Sudoku puzzles.

<br/>

## TensorFlow Configuration
This project is configured to use TensorFlow with CPU support. 

If you have a compatible GPU and want to leverage GPU acceleration, you can configure TensorFlow to use the GPU. Refer to the TensorFlow documentation for instructions on GPU setup.

### Checking Device Configuration
You can verify the TensorFlow device configuration on your system by running the provided Python script. If it shows "Currently used device: /CPU:0," TensorFlow is using the CPU.

```python
import tensorflow as tf

# Check the currently used device
print("Currently used device:", tf.test.gpu_device_name() or "/CPU:0")
```

<br/>

## Algorithm Overview
The Sudoku solver algorithm follows these key steps:

1. Input a Sudoku puzzle, typically represented as a 9x9 grid with some initial numbers.
2. Apply a backtracking algorithm to recursively fill in the empty cells with valid numbers.
3. Check for the validity of each number placement by ensuring it doesn't violate Sudoku rules.
4. Repeat the process until a solution is found or determine that no valid solution exists.

<br/>

## Set-Up and Run
### 1. Prerequisites
Before you begin, ensure that you have the following prerequisites installed on your system:

- **Python 3.x**
- **OpenCV (cv2)**
- **Numpy**
- **TensorFlow** (for loading the model)

### 2. Installation
Clone or download this GitHub repository to your local machine.

To clone the repository, open your terminal and use the following command:
   ```bash
   git clone https://github.com/aqib-qadir/Sudoku-Solver.git
   cd Sudoku-Solver
   ```

### 3. Usage
Open a terminal window and navigate to the project directory where you cloned the repository. You can change the Sudoku problem, and then run the script to get the output. 

<br/>

## Result
The result of running the Sudoku solver script will display the solved Sudoku grid on the console. 

`Note : Since we are using tensorflow, so it will take a while to show the output.`

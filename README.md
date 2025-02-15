A single runnable file for LLM training. Everything you need can be executed from the 1 file. Pick your method of training from the file name. Fill in all the places in the file with what you need to train, and run "Python *File_Name.py*" from a command prompt open in the same directory as the file.

As long as you have some sort of graphics card and train a model that fits in your VRAM, the training should work well.

It's as simple as that.


For better Lora training. Use my method bellow

# Continuous Fine-tuning Without Loss Using Lora and Mergekit

https://docs.google.com/document/d/1OjbjU5AOz4Ftn9xHQrX3oFQGhQ6RDUuXQipnQ9gn6tU/edit?usp=sharing


UPDATE FOR RUNNING "Galore_8bit_Version-2.py"


---
license: apache-2.0
---
Prerequisites:
1. Python:
https://www.python.org/downloads/
2. Git:
https://git-scm.com/downloads

Instructions:
1. Make sure python and git are installed
2. Open a command prompt terminal on your local folder
3. In command prompt run 
```
git lfs install
```
then
```
git clone https://huggingface.co/datasets/Rombo-Org/Easy_Galore_8bit_training_With_Native_Windows_Support
```
then
```
cd Easy_Galore_8bit_training_With_Native_Windows_Support
```
4. Now minimize the command prompt window and open the "Galore_8bit_Version-2.py" file
5. Edit the paramaters to suite your needs
6. Open the command prompt window and run
```
Python Galore_8bit_Version-2.py
```

The training will now run completely and save your model in the specified folder location. 


____________________________________________________________

# LATEST UPDATE

I added a new folder where you can train models with the max reduced Vram using Qlora and Galore. Download the folder, edit the config file, and run this command to execute it
```
python QaloreTraining.py --config_file config.txt
```

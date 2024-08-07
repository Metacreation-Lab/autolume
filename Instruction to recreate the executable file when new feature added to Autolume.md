**To recreate the executable file when new feature added to Autolume:**

* When adding new features, always note any new dependencies not already required by Autolume.  
* Ensure Autolume is developed in a virtual environment containing only its dependencies.


* ‘pack.bat’ is the script builds an executable from the \`main.spec\` file using PyInstaller, copies the \`assets\`, \`sr\_models\`, and \`models\` directories into the \`dist\\main\` directory, and then pauses to display a "packing finished" message.  
* Once a new feature is added and tested, check for any new files or folders required for Autolume's operations.  
* Adjust the paths in the script to your local path where the file is saved  
*  Add the necessary \`xcopy\` command(s) for these files/folders.  
* Run the \`pack.bat\` script from the root directory of the \`autolumelive\_colab\` project using the terminal: ‘pack.bat’

* This will build a new executable file with the existing \`main.spec\` script using PyInstaller.

* After the build process completes, navigate to \`\\root\_dir\\dist\\main\`.

* Run the executable file from the terminal.

* Test to ensure the program runs correctly and all features work as expected.

*   \- If the program does not run correctly or features do not work, review and update the \`main.spec\` script as needed.


Updating the PyInstaller Spec Script:

* Adjust the paths in the script to your local path where the file is saved  
* Structure of the Spec Script  
1. **Analysis**: Specifies the Python scripts, binaries, data files, and hidden imports needed.  
2. **PYZ**: Creates a Python archive with all the pure Python files.  
3. **EXE**: Bundles the Python archive and other components into an executable.  
4. **COLLECT**: Collects all files into a single directory.

* ## Adding New Binaries:

1. Locate the `binaries` section in the `Analysis` block.  
2. Add a tuple with the path to the binary file and the destination directory within the package.

* ## Adding New Data Files:

1. Locate the `datas` section in the `Analysis` block.  
2. Add a tuple with the path to the data file or directory and the destination directory within the package.

* ## Adding Hidden Imports:

1. Locate the `hiddenimports` section in the `Analysis` block.  
2. Add the module name to the list.

* ## Changing the Icon/name:

1. Locate the `EXE` block.  
2. Update the `icon or name` parameter with the path to the new icon file or new name.  
* Make sure to change the path.

More information about Pyinstaller, please see: [PyInstaller Manual](https://pyinstaller.org/en/stable/)



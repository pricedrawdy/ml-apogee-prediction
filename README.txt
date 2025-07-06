===============================
RocketSerializer Quick Guide
===============================

To convert an OpenRocket .ork file to JSON using RocketSerializer:

-------------
STEP 1: Open Terminal
-------------
Open Command Prompt (or PowerShell) and navigate to your RocketPy folder:

    cd C:\Users\price\Documents\RocketPy

-------------
STEP 2: Activate Virtual Environment
-------------
Activate the Python virtual environment:

    .\venv\Scripts\activate

-------------
STEP 3: Set JAVA_HOME (only needed if not already set globally)
-------------
If JAVA_HOME is not set permanently on your system, run:

    set JAVA_HOME=C:\Program Files\Java\jdk-17
    set PATH=%JAVA_HOME%\bin;%PATH%

-------------
STEP 4: Run RocketSerializer
-------------
Convert your .ork file to JSON using the following command:

    ork2json --filepath Trinity9.4.ork --ork_jar "C:/Program Files/OpenRocket/OpenRocket.jar" --output ./json_output

This will generate a JSON version of your `.ork` file and place it inside the `json_output` folder (which will be created if it doesn't exist).

-------------
NOTES:
-------------
- Make sure `OpenRocket.jar` exists at the specified path.
- You can replace `Trinity9.4.ork` with any other `.ork` file in the folder.
- You must be connected to the internet the first time you run RocketSerializer (for any dependency downloads).

===============================

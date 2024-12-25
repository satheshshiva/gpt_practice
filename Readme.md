## Setting Up the Virtual Environment
1. Create a virtual environment:
```
 python3 -m venv venv
 source venv/bin/activate
 python3 -m pip install -r requirements.txt
```
2. To activate the virtual environment everytime you run the project, use the command,
```
 source venv/bin/activate
 ```
3. Use the pip freeze command, if you have updated dependecies and if you to write them to `requirements.txt`
```
python3 -m pip freeze > requirements.txt
```
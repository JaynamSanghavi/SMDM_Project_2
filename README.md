# SMDM Project 2 - SEMANTIC ANALYSIS ON MOVIE REVIEWS

Our application will analyzing user reviews on movies and determine whether the sentiment expressed is positive or negative.

Steps for installation:
1. Pull the github repositry. 
```bash
git pull https://github.com/JaynamSanghavi/SMDM_Project_2
```

2. Goto backend folder and open cmd
3. Run the following command to install all the packages
```bash
pip install -r requirements.txt
```
4. Run the following command:
```bash
python app.py
```
5. Open postman or any api testing tool.
6. Test the route http://127.0.0.1:5000 and check if hello world is displaying or not.
7. Select the POST method and hit this url http://127.0.0.1:5000/predictComment. In body, select form-data and in key give this "comments" and the value would be the movie review.



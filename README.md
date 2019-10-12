Flower app is now deployed on Heroku!  
To access, visit: http://flower102-app-alanchn31.herokuapp.com  
Upload a flower image and a plot of the top 5 classes classified is shown

Flower app can also be run by pulling Docker Image through command:  
**"docker run -p 5000:5000 alanchn31/flower_class:version1"***

Docker commands:  
1) Build the app:
**"docker build --tag=flower_class ."**

2) Check machine’s local Docker image registry  
**"docker image ls"**  

3) Run the app, mapping your machine’s port 4000 to the container’s published port 80 using -p:  
**"docker run -p 5000:5000 flower_class"**

4) Share the Docker Image:  
i) Login   
**"docker login"**

ii) Tag the Image:
*docker tag image username/repository:tag*
In this case:  
**"docker tag flower_class alanchn31/flower_class:version1"**

iii) Push Image:  
**"docker push alanchn31/flower_class:version1"**

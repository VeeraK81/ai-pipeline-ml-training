- Machine Learning flow pipeline deployed in Jenkins as pipeline project.

- This container is running inside the Jenkins docker.  To know about this: 

Steps:
- Changes push to github
- Create pipeline in jenkins
- create env variables in jenkins
    ARTIFACT_STORE_URI=<bucket name>
    AWS_ACCESS_KEY_ID=<access key>
    AWS_SECRET_ACCESS_KEY=<secret>
    BACKEND_STORE_URI=<database url>
    PORT=7860
    MLFLOW_TRACKING_URI=<ml flow server url>
    

- go to jenkins in docker 
$ docker exec -it <jenkinsContainerId> bash
- navigate to /var/jenkins_home/workspace
$ docker ps
$ docker exec -it <containerId> bash

- to see the files and build details, navigate to /var/jenkins_home/jobs





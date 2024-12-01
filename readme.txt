- Machine Learning flow pipeline deployed in Jenkins as pipeline project.

- This container is running inside the Jenkins docker.  To know about this: 

Steps:
- Changes push to github
- Create pipeline in jenkins
- create env variables in jenkins 

- go to jenkins in docker 
$ docker exec -it <jenkinsContainerId> bash
- navigate to /var/jenkins_home/workspace
$ docker ps
$ docker exec -it <containerId> bash

- to see the files and build details, navigate to /var/jenkins_home/jobs




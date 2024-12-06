pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                // Checkout the code from the repository
                git branch: 'main', url: 'https://github.com/VeeraK81/ai-pipeline-ml-training.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    // Build the Docker image using the Dockerfile
                    sh 'docker build -t air-quality-pipeline-image .'
                }
            }
        }


        stage('Run Tests Inside Docker Container') {
            steps {
                withCredentials([
                    string(credentialsId: 'mlflow-tracking-uri', variable: 'MLFLOW_TRACKING_URI'),
                    string(credentialsId: 'aws-access-key', variable: 'AWS_ACCESS_KEY_ID'),
                    string(credentialsId: 'aws-secret-key', variable: 'AWS_SECRET_ACCESS_KEY'),
                    string(credentialsId: 'backend-store-uri', variable: 'BACKEND_STORE_URI'),
                    string(credentialsId: 'artifact-root', variable: 'ARTIFACT_ROOT'),
                    string(credentialsId: 'ai-solution-bucket-name', variable: 'BUCKET_NAME'),
                    string(credentialsId: 'ai-solution-file-key', variable: 'FILE_KEY')
                ]) {
                    
                    script {
                        sh '''
                            docker run \
                                -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
                                -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
                                -e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI \
                                -e ARTIFACT_ROOT=$ARTIFACT_ROOT \
                                -e BUCKET_NAME=$BUCKET_NAME \
                                -e FILE_KEY=$FILE_KEY \
                                -e BACKEND_STORE_URI=$BACKEND_STORE_URI \
                                air-quality-pipeline-image \
                                bash -c "pytest --maxfail=1 --disable-warnings"
                        '''
                    }
                }
            }
        }
    }

    post {
        always {
            // Clean up workspace and remove dangling Docker images
            sh 'docker system prune -f'
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed. Check logs for errors.'
        }
    }
}


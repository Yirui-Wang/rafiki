import abc
import os

class ContainerManager(abc.ABC):
    def __init__(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def create_service(self, service_name, image_name, replicas, args, environment_vars):
        '''
            Creates a service

            Args
                service_name: String - Name of the service
                image_name: String - Name of the Docker image to create a service for
                replicas: Int - Number of replicas to initialize for the service
                args: [String] - Arguments to pass to the service
                environment_vars: {String: String} - Dict of environment variable names to values

            Returns 
                id: String - ID for the service created
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def update_service(self, service_id, replicas):
        '''
            Updates the service's properties e.g. scaling the number of replicas

            Args
                service_id: String - ID of service to update
                repliaces: Int - Adjusted number of replicas for the service
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def destroy_service(self, service_id):
        '''
            Stops & destroys a service

            Args
                service_id: String - ID of service to destroy
        '''
        raise NotImplementedError()
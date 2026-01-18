import os
import numpy as np

class ModelPersistence:
      def __init__(self , filepath):
          self.filepath = filepath


      def save_model(self , model , weights=True , biases=True):
            if not os.path.exists(os.path.dirname(self.filepath)):
                os.makedirs(os.path.dirname(self.filepath))
           
            for i, layer in enumerate(self.model.layers):
               if hasattr(layer , 'weights') and weights:
                   np.savez_compressed(f"{self.filepath}_layer{i}_weights.npz" , layer.weights)
               if hasattr(layer , 'biases') and biases:
                   np.savez_compressed(f"{self.filepath}_layer{i}_biases.npz" , layer.biases)


      def load_model(self , model , weights=True , biases=True):
        for i, layer in enumerate(model.layers):
            if hasattr(layer , 'weights') and weights:
                data = np.load(f"{self.filepath}_layer{i}_weights.npz")
                layer.weights = data['arr_0']
            if hasattr(layer , 'biases') and biases:
                data = np.load(f"{self.filepath}_layer{i}_biases.npz")
                layer.biases = data['arr_0']    
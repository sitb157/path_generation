import tensorflow as tf
from tensorflow.keras import layers, models
def create_context_module(input_tensor):
    # Define the dilation rates for each layer
    dilation_rates = [(1, 1), (1, 1), (1, 1), (2, 4), (4, 8), (8, 12), (12, 16),
                                  (16, 20), (20, 24), (24, 28), (28, 32), (32, 1), (1, 1)]
    
    # Input 
    x = input_tensor
    
    for i, dilation in enumerate(dilation_rates):
        # Add dilated convolution layers
        x = layers.Conv2D(96, (3, 3), padding='same', dilation_rate=dilation)(x)
        x = layers.ELU()(x)
                
        # Add spatial dropout to all but the last layer
        if i < 12:
            x = layers.SpatialDropout2D(0.20)(x)
        
        # Reduce the number of feature maps in the final layer
        x = layers.Conv2D(16, (3, 3), padding='same', dilation_rate=dilation_rates[-1])(x)
        x = layers.ELU()(x)
    
    # Return the model
    return x

def create_model():
    inputs = layers.Input(shape=(600, 600, 8))

    # 1st and 2nd Convolutional layers
    x = layers.Conv2D(8, (3, 3), strides=(1, 1), padding='same')(inputs)
    x = layers.ELU()(x)
    x = layers.Conv2D(8, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.ELU()(x)
    
    # First Max Pooling
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # 3rd, 4th, and 5th Convolutional layers
    for _ in range(3):
        x = layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same')(x)
        x = layers.ELU()(x)
    
    # Second Max Pooling
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # Context Module (implement as described in Table 1)
    x = create_context_module(x)
    
    # Max Unpooling
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    
    # 10th, 11th, and 12th Convolutional layers
    for filters in [16, 16, 8]:
        x = layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(x)
        x = layers.ELU()(x)
    
    # Second Max Unpooling
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    
    # Final Convolutional layers
    x = layers.Conv2D(8, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(8, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(2, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.ELU()(x)
    
    # Softmax
    outputs = layers.Softmax()(x)
    model = models.Model(inputs=inputs, outputs=outputs)

    # Print model summary
    model.summary()
    return model

if __name__ == "__main__":
    model = create_model()

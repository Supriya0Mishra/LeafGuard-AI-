import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('model/leafguard_model.h5')

# Print model summary
print("\n✅ MODEL SUMMARY:\n")
model.summary()

# Check input shape
input_shape = model.input_shape
output_shape = model.output_shape

print(f"\n📐 Input shape: {input_shape}")
print(f"🎯 Output shape: {output_shape}")

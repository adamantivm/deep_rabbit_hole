use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{conv2d, linear, Conv2d, Linear, Module, VarBuilder};

// Define the CNN model architecture
struct MnistModel {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
}

impl MnistModel {
    fn new(vb: VarBuilder) -> Result<Self> {
        let conv1 = conv2d(1, 32, 5, Default::default(), vb.pp("conv1"))?;
        let conv2 = conv2d(32, 64, 5, Default::default(), vb.pp("conv2"))?;
        let fc1 = linear(1024, 1024, vb.pp("fc1"))?; // 64 * 4 * 4 = 1024 after two max_pools
        let fc2 = linear(1024, 10, vb.pp("fc2"))?;
        Ok(Self {
            conv1,
            conv2,
            fc1,
            fc2,
        })
    }
}

impl Module for MnistModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.conv1)?.relu()?;
        let xs = xs.max_pool2d(2)?;
        let xs = xs.apply(&self.conv2)?.relu()?;
        let xs = xs.max_pool2d(2)?;
        let xs = xs.flatten_from(1)?; // Flatten starting from dimension 1
        let xs = xs.apply(&self.fc1)?.relu()?;
        xs.apply(&self.fc2)
    }
}

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu; // Or Device::cuda_if_available(0)? for GPU

    // In a real scenario, you would load your pre-trained weights here.
    // For this example, we'll initialize with zero weights.
    // you'd load a `.safetensors` or other format file.
    // Example: let vb = VarBuilder::from_safetensors("model.safetensors", DType::F32, &device)?;
    println!("Initializing model with zero weights (replace with loading pre-trained weights in a real scenario).");
    let vb = VarBuilder::zeros(DType::F32, &device);
    let model = MnistModel::new(vb)?;

    // Create a dummy input tensor for an MNIST image (1 channel, 28x28 pixels)
    // The batch size is 1 for a single inference.
    let dummy_input = Tensor::randn(0f32, 1.0, (1, 1, 28, 28), &device)?;

    // Perform inference
    let logits = model.forward(&dummy_input)?;

    // Get the predicted class (the digit with the highest logit)
    let prediction = logits.argmax(1)?; // Argmax along dimension 1 (classes)
    let predicted_digit: u32 = prediction.i(0)?.to_scalar()?;

    println!("Dummy input shape: {:?}", dummy_input.shape());
    println!("Output logits shape: {:?}", logits.shape());
    println!("Predicted digit: {}", predicted_digit);

    Ok(())
}

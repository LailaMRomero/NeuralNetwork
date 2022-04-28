import java.util.*;
public class Layer {
	public Neuron[] neurons;
   public double[] weights;
	
	// Constructor for the hidden and output layer
	public Layer(int inNeurons,int numberNeurons) {
		this.neurons = new Neuron[numberNeurons];
		for(int i = 0; i < numberNeurons; i++) {
			 this.weights = new double[inNeurons]; //change to 2D array 9.23.20
			for(int j = 0; j < inNeurons ; j++) {
				weights[j] = Equations.RandomFloat(Neuron.minWeightValue, Neuron.maxWeightValue);

			}
          neurons[i] = new Neuron(weights,Equations.RandomFloat(0, 1));

         }

         }

	
	// Constructor for the input layer
	public Layer(float input[]) {
		this.neurons = new Neuron[input.length];
		for(int i = 0; i < input.length; i++) {
			this.neurons[i] = new Neuron(input[i]);
		}
	}
}

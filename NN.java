public class NN{
 
    int Layers;
    int inNodes;
    int hiddenNodes;
    int outNodes;
    Layer[] layers; 
    static TrainingData[] tDataSet; 
    static TrainingData[] tDataSet2; 

   
    public NN( int inNodes, int hiddenNodes, int outNodes) {
        this.inNodes = inNodes;
        this.hiddenNodes = hiddenNodes;
        this.outNodes=outNodes;
        createNetwork( inNodes, hiddenNodes, outNodes);
    }
    
    
   public void createNetwork( int inNodes, int hiddenNodes, int outNodes){
      //if you wish to add or remove a layer you must change the size of the array and either add/remove an index
      // ex: to add a 2nd hidden layer you change the Layer size and add the following:     	
      //layers[2] = new Layer(inNodes,hiddenNodes); 
      //the last layer should be of index 3 in this scenario

      layers = new Layer[3];
    	layers[0] = null; // Input layers are null because they don't have any nodes coming into them
    	layers[1] = new Layer(inNodes,hiddenNodes); // Inital Hidden Layer 
    	layers[2] = new Layer(hiddenNodes,outNodes); // Initial Output Layer 
      System.out.println("You created a neural network of " +layers.length + " layers\nThere are " +inNodes + " input nodes in the input layer, "
       + hiddenNodes +" hidden nodes in each hidden layer, and " + outNodes +" output nodes in the output layer.\n");
      CreateTrainingData(inNodes, outNodes);
      }
      

      // creates 3 training data sets of all 0's, all 1's and alternating 0's and 1's
   public static void CreateTrainingData(float inputNodes, float outNodes) {
        // type-casting from float to int
        int inputNodes1 = (int) inputNodes;        
        float[] input1 = new float[inputNodes1]; 
        float[] input2 = new float[inputNodes1]; 
        float[] input3 = new float[inputNodes1]; 
       
       
         // here I am declaring the expected outputs for each output node and data set      
        float[] expectedOutput1 = new float[] {0,1}; //for dataset1 I expect the output nodes to show 0 and 1
        float[] expectedOutput2 = new float[] {1,1}; // for dataset2 I expect the output nodes to show 1 and 1
        float[] expectedOutput3 = new float[] {1,0}; // for dataset3 I expect the output nodes to show 1 and 0
      
       //sets inputs equal to 0
       for(int i=0; i <inputNodes; i++){
               input1[i]=0;
        }
        
       //sets inputs equal to 1
       for(int i=0; i <inputNodes; i++){
               input2[i]=1;
        }
        
      // sets aternating values of 0 and 1
      for(int i =0; i<inputNodes; i++){
         if(i%2==0){
            input3[i]=0;
         }else{
            input3[i]=1;
            }
      }
      
        // My changes (using an array for the data sets)
        tDataSet = new TrainingData[3];
        tDataSet[0] = new TrainingData(input1, expectedOutput1);
        tDataSet[1] = new TrainingData(input2, expectedOutput2);
        tDataSet[2] = new TrainingData(input3, expectedOutput3);
              
               
    }
   public void forward(float[] inputs) {
    	// First bring the inputs into the input layer layers[0]
    	  layers[0] = new Layer(inputs);
    	
        for(int i = 1; i < layers.length; i++) {
        	for(int j = 0; j < layers[i].neurons.length; j++) {
        		float sum = 0;
        		for(int k = 0; k < layers[i-1].neurons.length; k++) {
        			sum += layers[i-1].neurons[k].value*layers[i].neurons[j].weights[k];
        		}
        		sum += layers[i].neurons[j].bias; // add in the bias 
        		layers[i].neurons[j].value = Equations.Sigmoid(sum);
        	}
        } 	
    }
    
        // Calculate the output layer weights, calculate the hidden layer weight then update all the weights
    public void backward(float learning_rate,TrainingData tData) {
    	
    	int number_layers = layers.length;
    	int out_index = number_layers-1;
    	
    	// Update the output layers 
    	// For each output
    	for(int i = 0; i < layers[out_index].neurons.length; i++) {
    		// and for each of their weights
    		float output = layers[out_index].neurons[i].value;
    		float target = tData.expectedOutput[i];
    		float derivative = output-target;
    		float delta = derivative*(output*(1-output));
    		layers[out_index].neurons[i].gradient = delta;
    		for(int j = 0; j < layers[out_index].neurons[i].weights.length;j++) { 
    			float previous_output = layers[out_index-1].neurons[j].value;
    			float error = delta*previous_output;
    			layers[out_index].neurons[i].cache_weights[j] = layers[out_index].neurons[i].weights[j] - learning_rate*error;
    		}
    	}
    	
    	//Update all the subsequent hidden layers
    	for(int i = out_index-1; i > 0; i--) {
    		// For all neurons in that layers
    		for(int j = 0; j < layers[i].neurons.length; j++) {
    			float output = layers[i].neurons[j].value;
    			float gradient_sum = sumGradient(j,i+1);
    			float delta = (gradient_sum)*(output*(1-output));
    			layers[i].neurons[j].gradient = delta;
    			// And for all their weights
    			for(int k = 0; k < layers[i].neurons[j].weights.length; k++) {
    				float previous_output = layers[i-1].neurons[k].value;
    				float error = delta*previous_output;
    				layers[i].neurons[j].cache_weights[k] = layers[i].neurons[j].weights[k] - learning_rate*error;
    			}
    		}
    	}
    	
    	// Here we do another pass where we update all the weights
    	for(int i = 0; i< layers.length;i++) {
    		for(int j = 0; j < layers[i].neurons.length;j++) {
    			layers[i].neurons[j].update_weight();

    		}
    	}
    	
    } 
    
   public float sumGradient(int n_index,int l_index) {
    	float gradient_sum = 0;
    	Layer current_layer = layers[l_index];
    	for(int i = 0; i < current_layer.neurons.length; i++) {
    		Neuron current_neuron = current_layer.neurons[i];
    		gradient_sum += current_neuron.weights[n_index]*current_neuron.gradient;
    	}
    	return gradient_sum;
    }
 
    
    // This function is used to train being forward and backward.
    public void train(int training_iterations,float learning_rate, TrainingData[] dataset) {
    	for(int i = 0; i < training_iterations; i++) {
    		for(int j = 0; j < tDataSet.length; j++) {
    			forward(tDataSet[j].data);
    			backward(learning_rate,tDataSet[j]);

    		}
    	}
    }


    }
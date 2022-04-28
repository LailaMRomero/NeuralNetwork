import java.util.*;
import java.math.RoundingMode;
import java.text.DecimalFormat;
public class NN_Main {


/* This is the driver class for the neural network. Here you initialize the creation of the networks
   Something to note: when accessing layers and nodes for the network all indexes start at 0, for example layer[0] is the input layer
   neuron[0] would be the first neuron in the specified layer
*/

    // Main Method
  public static void main(String[] args) {
  
      // this is the creation of a neural network called nn1
	   // 2 input nodes, 3 hidden nodes, 2 output nodes
       NN nn1 = new NN(2,3,2);
       
       
       // this is used to round up numbers to 8 decimal places 
       DecimalFormat df = new DecimalFormat("#.########");
       df.setRoundingMode(RoundingMode.CEILING);
       
        // Set the Min and Max weight value for all Neurons
       Neuron.setRangeWeight(-1, 1);
       
       //these are the expected outputs for all 3 sets of data
       // if are multiple outputs then the expected should match
       // for example there are 3 sets of data and there are 2 output nodes so there should be 6 expected outputs
       float[] expectedOutputs = new float [] {0,1,1,1,1,1};
       
       

        System.out.println("*******************************");
        System.out.println("Initial weights before training");
        System.out.println("*******************************\n");
        System.out.println(Arrays.toString(nn1.layers[1].neurons[0].weights));
        System.out.println(Arrays.toString(nn1.layers[1].neurons[1].weights));
        System.out.println(Arrays.toString(nn1.layers[1].neurons[2].weights));
        System.out.println(Arrays.toString(nn1.layers[2].neurons[0].weights));
        System.out.println(Arrays.toString(nn1.layers[2].neurons[1].weights));

        System.out.println("\n*********************");
        System.out.println("Output before training");
        System.out.println("***********************");

        
        
        for(int i = 0; i < nn1.tDataSet.length; i++) {
            nn1.forward(nn1.tDataSet[i].data);
            System.out.println("\nNN1 first output node: " +nn1.layers[2].neurons[0].value);
            System.out.println("NN2 second output node: " + nn1.layers[2].neurons[1].value);
        }
       
        
        
        //starts a timer to time how long the training takes
        double startTime = System.nanoTime();
        nn1.train(1000000, 0.05f, nn1.tDataSet);

        double endTime = System.nanoTime();
        endTime = (endTime-startTime)/1000000; 
        
        //this is the array for the expected outputs, as of right now it is set to 6 
        //because there 3 sets of training data and 2 outputs for each set
        
        float [] actualOutputs = new float [6];
        
        System.out.println("\n********************");
        System.out.println("Output after training");
        System.out.println("********************");


        for(int i = 0; i < nn1.tDataSet.length; i++) {
            nn1.forward(nn1.tDataSet[i].data);
            actualOutputs[i] =nn1.layers[2].neurons[0].value;
            System.out.println("\nNN1 first output node: "+ nn1.layers[2].neurons[0].value); // this will display the values for the first node in the output layer
            System.out.println("NN1 second output node: "+ nn1.layers[2].neurons[1].value); // this will display the values for the second node in the output layer
        }
                                            
        System.out.println("\n*******************");
        System.out.println("Updated weights NN1:");
        System.out.println("**********************\n");
        
        System.out.println(Arrays.toString(nn1.layers[1].neurons[0].weights));
        System.out.println(Arrays.toString(nn1.layers[1].neurons[1].weights));
        System.out.println(Arrays.toString(nn1.layers[1].neurons[2].weights));
        System.out.println(Arrays.toString(nn1.layers[2].neurons[0].weights));
        System.out.println(Arrays.toString(nn1.layers[2].neurons[1].weights));
        
        System.out.println("\n****************************");
        System.out.println("Time taken to train in seconds: " + endTime/1000);
        System.out.println("*******************************\n");

        float error1 = Equations.sumSquaredError(actualOutputs, expectedOutputs, 3); 
        System.out.println("Mean Squared Error: " + df.format(Equations.sumSquaredError(actualOutputs, expectedOutputs, 3)));

    }
    
    }

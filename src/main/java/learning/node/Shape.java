package learning.node;

import learning.tensor.Tensor;
import learning.optimize.Optimizer;
import learning.operation.Operation;

import org.jocl.*;
import static org.jocl.CL.*;
import static learning.ComputeContext.*;

public class Shape extends Node {
   
   public Shape (int[] size) {
      outputSize = size;
   }
   
   void forwardProp() {
      if(OPEN_CL) {
         output = new Tensor(in, outputSize);
      } else {
         
      }
   }
   
   void backwardProp() {
      if(OPEN_CL) {
         input = new Tensor(dO, before.getOutputSize());
      } else {
         
      }
   }
   
   protected void setOutputSize() {
      int bufferSize = 1;
      for(int i = 0; i<outputSize.length; i++) {
         bufferSize *= outputSize[i];
      }
      
      int bufferSize2 = 1;
      for(int i = 0; i<before.getOutputSize().length; i++) {
         bufferSize2 *= before.getOutputSize()[i];
      }
      
      if(bufferSize != bufferSize2) {
         System.err.println("Reshaping with different buffer sizes");
      }
   }
   
}
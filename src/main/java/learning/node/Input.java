package learning.node;

import learning.tensor.Tensor;
import learning.operation.Operation;

public class Input extends Node {
   
   public Input (int[] size) {
      outputSize = size;
   }
   
   @Override
   void forwardProp() {
      
   }
   
   @Override
   void backwardProp() {
      input = dO;
   }
   
   @Override
   Tensor forwards(Operation op) {
      if(op.calculate) forwardProp();
      op.operate(this,output);
      return output;
   }
   
   @Override
   void backwards(Tensor t, Operation op) {
      dO = t;
      if(op.calculate) backwardProp();
      op.operate(this,input);
   }
   
   @Override
   protected void attachBefore(Node n) {
      //should not even be called in the first place
   }
   
   public void load(Tensor in) {
      output = in;
   }
   
}
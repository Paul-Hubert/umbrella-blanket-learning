package learning.node;

import learning.tensor.Tensor;
import learning.operation.Operation;

public class Input extends Node {
   
   public Input (int[] size) {
      super();
      outputSize = size;
   }
   
   @Override
   public void forwardProp() {
      
   }
   
   @Override
   public void backwardProp() {
      input = dO;
   }
   
   @Override
   Tensor forwards() {
      net.operate(this);
      return output;
   }
   
   @Override
   void backwards(Tensor t) {
      dO = t;
      net.operate(this);
   }
   
   @Override
   protected void attachBefore(Node n) {
      //should not even be called in the first place
   }
   
   public void load(Tensor in) {
      output = in;
   }
   
}
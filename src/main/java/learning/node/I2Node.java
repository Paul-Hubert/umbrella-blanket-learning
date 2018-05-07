package learning.node;

import learning.tensor.Tensor;
import learning.operation.Operation;

public class I2Node extends Node {
   
   protected Node before2;
   
   protected Tensor input2, in2;
   
   public void forwardProp() {
      output = in;
   }
   
   public void backwardProp() {
      input = dO;
      input2 = dO;
   }
   
   Tensor forwards() {
      in = before.forwards();
      in2 = before2.forwards();
      net.operate(this);
      return output;
   }
   
   void backwards(Tensor t) {
      dO = t;
      net.operate(this);
      before.backwards(input);
      before2.backwards(input2);
   }
   
   protected void attachBefore(Node n) {
      if(before == null) {
         before = n;
         return;
      }
      before2 = n;
      
      setOutputSize();
      prepare();
   }
}
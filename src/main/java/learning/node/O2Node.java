package learning.node;

import learning.tensor.Tensor;
import learning.operation.Operation;

public class O2Node extends Node {
   
   protected Node next2;
   
   protected Tensor output2, dO2;
   
   protected boolean calculated = false;
   
   public void forwardProp() {
      output = in;
      output2 = in;
   }
   
   public void backwardProp() {
      input = dO;
   }
   
   Tensor forwards() {
      if(!calculated) {
         in = before.forwards();
         calculated = true;
      }
      if(net.getLast() == next) {
         net.operate(this);
         return output;
      } if(net.getLast() == next2) {
         net.operate(this);
         return output2;
      }
      return null;
   }
   
   void backwards(Tensor t) {
      dO = t;
      net.operate(this);
      before.backwards(input);
   }
   
   protected void attachNext(Node n) {
      if(next==null) {
         next = n;
      } else {
         next2 = n;
      }
   }
   
}